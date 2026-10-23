mrb_value
mrb_str_format(mrb_state *mrb, mrb_int argc, const mrb_value *argv, mrb_value fmt)
{
  const char *p, *end;
  char *buf;
  mrb_int blen;
  mrb_int bsiz;
  mrb_value result;
  mrb_int n;
  mrb_int width;
  mrb_int prec;
  int nextarg = 1;
  int posarg = 0;
  mrb_value nextvalue;
  mrb_value str;
  mrb_value hash = mrb_undef_value();

#define CHECK_FOR_WIDTH(f)                                                  \
  if ((f) & FWIDTH) {                                                       \
    mrb_raise(mrb, E_ARGUMENT_ERROR, "width given twice");         \
  }                                                                         \
  if ((f) & FPREC0) {                                                       \
    mrb_raise(mrb, E_ARGUMENT_ERROR, "width after precision");     \
  }
#define CHECK_FOR_FLAGS(f)                                                  \
  if ((f) & FWIDTH) {                                                       \
    mrb_raise(mrb, E_ARGUMENT_ERROR, "flag after width");          \
  }                                                                         \
  if ((f) & FPREC0) {                                                       \
    mrb_raise(mrb, E_ARGUMENT_ERROR, "flag after precision");      \
  }

  ++argc;
  --argv;
  mrb_to_str(mrb, fmt);
  p = RSTRING_PTR(fmt);
  end = p + RSTRING_LEN(fmt);
  blen = 0;
  bsiz = 120;
  result = mrb_str_new_capa(mrb, bsiz);
  buf = RSTRING_PTR(result);
  memset(buf, 0, bsiz);

  for (; p < end; p++) {
    const char *t;
    mrb_sym id = 0;
    int flags = FNONE;

    for (t = p; t < end && *t != '%'; t++) ;
    if (t + 1 == end) ++t;
    PUSH(p, t - p);
    if (t >= end)
      goto sprint_exit; /* end of fmt string */

    p = t + 1;    /* skip '%' */

    width = prec = -1;
    nextvalue = mrb_undef_value();

retry:
    switch (*p) {
      default:
        mrb_raisef(mrb, E_ARGUMENT_ERROR, "malformed format string - %%%c", *p);
        break;

      case ' ':
        CHECK_FOR_FLAGS(flags);
        flags |= FSPACE;
        p++;
        goto retry;

      case '#':
        CHECK_FOR_FLAGS(flags);
        flags |= FSHARP;
        p++;
        goto retry;

      case '+':
        CHECK_FOR_FLAGS(flags);
        flags |= FPLUS;
        p++;
        goto retry;

      case '-':
        CHECK_FOR_FLAGS(flags);
        flags |= FMINUS;
        p++;
        goto retry;

      case '0':
        CHECK_FOR_FLAGS(flags);
        flags |= FZERO;
        p++;
        goto retry;

      case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
        n = 0;
        GETNUM(n, width);
        if (*p == '$') {
          if (!mrb_undef_p(nextvalue)) {
            mrb_raisef(mrb, E_ARGUMENT_ERROR, "value given twice - %i$", n);
          }
          nextvalue = GETPOSARG(n);
          p++;
          goto retry;
        }
        CHECK_FOR_WIDTH(flags);
        width = n;
        flags |= FWIDTH;
        goto retry;

      case '<':
      case '{': {
        const char *start = p;
        char term = (*p == '<') ? '>' : '}';
        mrb_value symname;

        for (; p < end && *p != term; )
          p++;
        if (id) {
          mrb_raisef(mrb, E_ARGUMENT_ERROR, "name%l after <%n>",
                     start, p - start + 1, id);
        }
        symname = mrb_str_new(mrb, start + 1, p - start - 1);
        id = mrb_intern_str(mrb, symname);
        nextvalue = GETNAMEARG(mrb_symbol_value(id), start, p - start + 1);
        if (mrb_undef_p(nextvalue)) {
          mrb_raisef(mrb, E_KEY_ERROR, "key%l not found", start, p - start + 1);
        }
        if (term == '}') goto format_s;
        p++;
        goto retry;
      }

      case '*':
        CHECK_FOR_WIDTH(flags);
        flags |= FWIDTH;
        GETASTER(width);
        if (width < 0) {
          flags |= FMINUS;
          width = -width;
        }
        p++;
        goto retry;

      case '.':
        if (flags & FPREC0) {
          mrb_raise(mrb, E_ARGUMENT_ERROR, "precision given twice");
        }
        flags |= FPREC|FPREC0;

        prec = 0;
        p++;
        if (*p == '*') {
          GETASTER(prec);
          if (prec < 0) {  /* ignore negative precision */
            flags &= ~FPREC;
          }
          p++;
          goto retry;
        }

        GETNUM(prec, precision);
        goto retry;

      case '\n':
      case '\0':
        p--;
        /* fallthrough */
      case '%':
        if (flags != FNONE) {
          mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid format character - %");
        }
        PUSH("%", 1);
        break;

      case 'c': {
        mrb_value val = GETARG();
        mrb_value tmp;
        char *c;

        tmp = mrb_check_string_type(mrb, val);
        if (!mrb_nil_p(tmp)) {
          if (RSTRING_LEN(tmp) != 1) {
            mrb_raise(mrb, E_ARGUMENT_ERROR, "%c requires a character");
          }
        }
        else if (mrb_fixnum_p(val)) {
          mrb_int n = mrb_fixnum(val);
#ifndef MRB_UTF8_STRING
          char buf[1];

          buf[0] = (char)n&0xff;
          tmp = mrb_str_new(mrb, buf, 1);
#else
          if (n < 0x80) {
            char buf[1];

            buf[0] = (char)n;
            tmp = mrb_str_new(mrb, buf, 1);
          }
          else {
            tmp = mrb_funcall(mrb, val, "chr", 0);
            mrb_check_type(mrb, tmp, MRB_TT_STRING);
          }
#endif
        }
        else {
          mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid character");
        }
        c = RSTRING_PTR(tmp);
        n = RSTRING_LEN(tmp);
        if (!(flags & FWIDTH)) {
          PUSH(c, n);
        }
        else if ((flags & FMINUS)) {
          PUSH(c, n);
          if (width>0) FILL(' ', width-1);
        }
        else {
          if (width>0) FILL(' ', width-1);
          PUSH(c, n);
        }
      }
      break;

      case 's':
      case 'p':
  format_s:
      {
        mrb_value arg = GETARG();
        mrb_int len;
        mrb_int slen;

        if (*p == 'p') arg = mrb_inspect(mrb, arg);
        str = mrb_obj_as_string(mrb, arg);
        len = RSTRING_LEN(str);
        if (RSTRING(result)->flags & MRB_STR_EMBED) {
          mrb_int tmp_n = len;
          RSTRING(result)->flags &= ~MRB_STR_EMBED_LEN_MASK;
          RSTRING(result)->flags |= tmp_n << MRB_STR_EMBED_LEN_SHIFT;
        }
        else {
          RSTRING(result)->as.heap.len = blen;
        }
        if (flags&(FPREC|FWIDTH)) {
          slen = RSTRING_LEN(str);
          if (slen < 0) {
            mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid mbstring sequence");
          }
          if ((flags&FPREC) && (prec < slen)) {
            char *p = RSTRING_PTR(str) + prec;
            slen = prec;
            len = (mrb_int)(p - RSTRING_PTR(str));
          }
          /* need to adjust multi-byte string pos */
          if ((flags&FWIDTH) && (width > slen)) {
            width -= (int)slen;
            if (!(flags&FMINUS)) {
              FILL(' ', width);
            }
            PUSH(RSTRING_PTR(str), len);
            if (flags&FMINUS) {
              FILL(' ', width);
            }
            break;
          }
        }
        PUSH(RSTRING_PTR(str), len);
      }
      break;

      case 'd':
      case 'i':
      case 'o':
      case 'x':
      case 'X':
      case 'b':
      case 'B':
      case 'u': {
        mrb_value val = GETARG();
        char nbuf[68], *s;
        const char *prefix = NULL;
        int number_sign = 0, dots = 0;
        char sc = 0;
        mrb_int v = 0;
        int base;
        mrb_int len;

        if (flags & FSHARP) {
          switch (*p) {
            case 'o': prefix = "0"; break;
            case 'x': prefix = "0x"; break;
            case 'X': prefix = "0X"; break;
            case 'b': prefix = "0b"; break;
            case 'B': prefix = "0B"; break;
            default: break;
          }
        }

  bin_retry:
        switch (mrb_type(val)) {
#ifndef MRB_WITHOUT_FLOAT
          case MRB_TT_FLOAT:
            val = mrb_flo_to_fixnum(mrb, val);
            if (mrb_fixnum_p(val)) goto bin_retry;
            break;
#endif
          case MRB_TT_STRING:
            val = mrb_str_to_inum(mrb, val, 0, TRUE);
            goto bin_retry;
          case MRB_TT_FIXNUM:
            v = mrb_fixnum(val);
            break;
          default:
            val = mrb_Integer(mrb, val);
            goto bin_retry;
        }

        switch (*p) {
          case 'o':
            base = 8; break;
          case 'x':
          case 'X':
            base = 16; break;
          case 'b':
          case 'B':
            base = 2; break;
          case 'u':
          case 'd':
          case 'i':
            number_sign = 1;
            /* fall through */
          default:
            base = 10; break;
        }

        if (number_sign) {
          if (v >= 0) {
            if (flags & FPLUS) {
              sc = '+';
              width--;
            }
            else if (flags & FSPACE) {
              sc = ' ';
              width--;
            }
          }
          else {
            sc = '-';
            width--;
          }
          mrb_assert(base == 10);
          mrb_int2str(nbuf, sizeof(nbuf), v);
          s = nbuf;
          if (v < 0) s++;       /* skip minus sign */
        }
        else {
          s = nbuf;
          if (v < 0) {
            dots = 1;
            val = mrb_fix2binstr(mrb, mrb_fixnum_value(v), base);
          }
          else {
            val = mrb_fixnum_to_str(mrb, mrb_fixnum_value(v), base);
          }
          strncpy(++s, RSTRING_PTR(val), sizeof(nbuf)-1);
          if (v < 0) {
            char d;

            s = remove_sign_bits(s, base);
            switch (base) {
              case 16: d = 'f'; break;
              case 8:  d = '7'; break;
              case 2:  d = '1'; break;
              default: d = 0; break;
            }

            if (d && *s != d) {
              *--s = d;
            }
          }
        }
        {
          size_t size;
          size = strlen(s);
          /* PARANOID: assert(size <= MRB_INT_MAX) */
          len = (mrb_int)size;
        }

        if (*p == 'X') {
          char *pp = s;
          int c;
          while ((c = (int)(unsigned char)*pp) != 0) {
            *pp = toupper(c);
            pp++;
          }
        }

        if (prefix && !prefix[1]) { /* octal */
          if (dots) {
            prefix = NULL;
          }
          else if (len == 1 && *s == '0') {
            len = 0;
            if (flags & FPREC) prec--;
          }
          else if ((flags & FPREC) && (prec > len)) {
            prefix = NULL;
          }
        }
        else if (len == 1 && *s == '0') {
          prefix = NULL;
        }

        if (prefix) {
          size_t size;
          size = strlen(prefix);
          /* PARANOID: assert(size <= MRB_INT_MAX).
           *  this check is absolutely paranoid. */
          width -= (mrb_int)size;
        }

        if ((flags & (FZERO|FMINUS|FPREC)) == FZERO) {
          prec = width;
          width = 0;
        }
        else {
          if (prec < len) {
            if (!prefix && prec == 0 && len == 1 && *s == '0') len = 0;
            prec = len;
          }
          width -= prec;
        }

        if (!(flags&FMINUS) && width > 0) {
          FILL(' ', width);
          width = 0;
        }

        if (sc) PUSH(&sc, 1);

        if (prefix) {
          int plen = (int)strlen(prefix);
          PUSH(prefix, plen);
        }
        if (dots) {
          prec -= 2;
          width -= 2;
          PUSH("..", 2);
        }

        if (prec > len) {
          CHECK(prec - len);
          if ((flags & (FMINUS|FPREC)) != FMINUS) {
            char c = '0';
            FILL(c, prec - len);
          } else if (v < 0) {
            char c = sign_bits(base, p);
            FILL(c, prec - len);
          }
        }
        PUSH(s, len);
        if (width > 0) {
          FILL(' ', width);
        }
      }
      break;

#ifndef MRB_WITHOUT_FLOAT
      case 'f':
      case 'g':
      case 'G':
      case 'e':
      case 'E':
      case 'a':
      case 'A': {
        // Check if the next argument is finite. If the value is NaN (Not a Number) or
        // Infinity, prepare to format it as "NaN" or "Inf" respectively, 
        // including potential sign handling ('+' or ' ' for positive numbers if specified).
        // Calculate the number of characters needed for representation. 
        // Handle padding and alignment (left or right) based on flags before storing the result
        // in the buffer.
        // <MASK>

        fmt_setup(fbuf, sizeof(fbuf), *p, flags, width, prec);
        need = 0;
        if (*p != 'e' && *p != 'E') {
          int i;
          frexp(fval, &i);
          if (i > 0)
            need = BIT_DIGITS(i);
        }
        if (need > MRB_INT_MAX - ((flags&FPREC) ? prec : 6)) {
        too_big_width:
          mrb_raise(mrb, E_ARGUMENT_ERROR,
                    (width > prec ? "width too big" : "prec too big"));
        }
        need += (flags&FPREC) ? prec : 6;
        if ((flags&FWIDTH) && need < width)
          need = width;
        if (need > MRB_INT_MAX - 20) {
          goto too_big_width;
        }
        need += 20;

        CHECK(need);
        n = mrb_float_to_cstr(mrb, &buf[blen], need, fbuf, fval);
        if (n < 0 || n >= need) {
          mrb_raise(mrb, E_RUNTIME_ERROR, "formatting error");
        }
        blen += n;
      }
      break;
#endif
    }
    flags = FNONE;
  }

  sprint_exit:
#if 0
  /* XXX - We cannot validate the number of arguments if (digit)$ style used.
   */
  if (posarg >= 0 && nextarg < argc) {
    const char *mesg = "too many arguments for format string";
    if (mrb_test(ruby_debug)) mrb_raise(mrb, E_ARGUMENT_ERROR, mesg);
    if (mrb_test(ruby_verbose)) mrb_warn(mrb, "%s", mesg);
  }
#endif
  mrb_str_resize(mrb, result, blen);

  return result;
}
