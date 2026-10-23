PHPAPI HashTable *php_fgetcsv(php_stream *csv_stream, char delimiter, char enclosure, int escape_char, size_t buf_len, char *buf) /* {{{ */
{
	char *temp, *bptr, *line_end, *limit;
	size_t temp_len, line_end_len;
	int inc_len;
	bool first_field = true;

	ZEND_ASSERT((escape_char >= 0 && escape_char <= UCHAR_MAX) || escape_char == PHP_CSV_NO_ESCAPE);

	/* initialize internal state */
	php_mb_reset();

	/* Now into new section that parses buf for delimiter/enclosure fields */

	/* Strip trailing space from buf, saving end of line in case required for enclosure field */

	bptr = buf;
	line_end = limit = (char *)php_fgetcsv_lookup_trailing_spaces(buf, buf_len);
	line_end_len = buf_len - (size_t)(limit - buf);

	/* reserve workspace for building each individual field */
	temp_len = buf_len;
	temp = emalloc(temp_len + line_end_len + 1);

	/* Initialize values HashTable */
	HashTable *values = zend_new_array(0);

	/* Main loop to read CSV fields */
	/* NB this routine will return NULL for a blank line */
	do {
		char *comp_end, *hunk_begin;
		char *tptr = temp;

		inc_len = (bptr < limit ? (*bptr == '\0' ? 1 : php_mblen(bptr, limit - bptr)): 0);
		if (inc_len == 1) {
			char *tmp = bptr;
			while ((*tmp != delimiter) && isspace((int)*(unsigned char *)tmp)) {
				tmp++;
			}
			if (*tmp == enclosure && tmp < limit) {
				bptr = tmp;
			}
		}

		if (first_field && bptr == line_end) {
			zend_array_destroy(values);
			values = NULL;
			break;
		}
		first_field = false;
		/* 2. Read field, leaving bptr pointing at start of next field */
		if (inc_len != 0 && *bptr == enclosure) {
			int state = 0;

			bptr++;	/* move on to first character in field */
			hunk_begin = bptr;

			/* 2A. handle enclosure delimited field */
			for (;;) {
				switch (inc_len) {
					case 0:
						switch (state) {
							case 2:
								memcpy(tptr, hunk_begin, bptr - hunk_begin - 1);
								tptr += (bptr - hunk_begin - 1);
								hunk_begin = bptr;
								goto quit_loop_2;

							case 1:
								memcpy(tptr, hunk_begin, bptr - hunk_begin);
								tptr += (bptr - hunk_begin);
								hunk_begin = bptr;
								ZEND_FALLTHROUGH;

							case 0: {
								if (hunk_begin != line_end) {
									memcpy(tptr, hunk_begin, bptr - hunk_begin);
									tptr += (bptr - hunk_begin);
									hunk_begin = bptr;
								}

								/* add the embedded line end to the field */
								memcpy(tptr, line_end, line_end_len);
								tptr += line_end_len;

								/* nothing can be fetched if stream is NULL (e.g. str_getcsv()) */
								if (csv_stream == NULL) {
									/* the enclosure is unterminated */
									if (bptr > limit) {
										/* if the line ends with enclosure, we need to go back by
										 * one character so the \0 character is not copied. */
										if (hunk_begin == bptr) {
											--hunk_begin;
										}
										--bptr;
									}
									goto quit_loop_2;
								}

								size_t new_len;
								char *new_buf = php_stream_get_line(csv_stream, NULL, 0, &new_len);
								if (!new_buf) {
									/* we've got an unterminated enclosure,
									 * assign all the data from the start of
									 * the enclosure to end of data to the
									 * last element */
									if (bptr > limit) {
										/* if the line ends with enclosure, we need to go back by
										 * one character so the \0 character is not copied. */
										if (hunk_begin == bptr) {
											--hunk_begin;
										}
										--bptr;
									}
									goto quit_loop_2;
								}

								temp_len += new_len;
								char *new_temp = erealloc(temp, temp_len);
								tptr = new_temp + (size_t)(tptr - temp);
								temp = new_temp;

								efree(buf);
								buf_len = new_len;
								bptr = buf = new_buf;
								hunk_begin = buf;

								line_end = limit = (char *)php_fgetcsv_lookup_trailing_spaces(buf, buf_len);
								line_end_len = buf_len - (size_t)(limit - buf);

								state = 0;
							} break;
						}
						break;

					case -2:
					case -1:
						php_mb_reset();
						ZEND_FALLTHROUGH;
					case 1:
						/* we need to determine if the enclosure is
						 * 'real' or is it escaped */
						switch (state) {
							case 1: /* escaped */
								bptr++;
								state = 0;
								break;
							case 2: /* embedded enclosure ? let's check it */
								if (*bptr != enclosure) {
									/* real enclosure */
									memcpy(tptr, hunk_begin, bptr - hunk_begin - 1);
									tptr += (bptr - hunk_begin - 1);
									hunk_begin = bptr;
									goto quit_loop_2;
								}
								memcpy(tptr, hunk_begin, bptr - hunk_begin);
								tptr += (bptr - hunk_begin);
								bptr++;
								hunk_begin = bptr;
								state = 0;
								break;
							default:
								if (*bptr == enclosure) {
									state = 2;
								} else if (escape_char != PHP_CSV_NO_ESCAPE && *bptr == escape_char) {
									state = 1;
								}
								bptr++;
								break;
						}
						break;

					default:
						switch (state) {
							case 2:
								/* real enclosure */
								memcpy(tptr, hunk_begin, bptr - hunk_begin - 1);
								tptr += (bptr - hunk_begin - 1);
								hunk_begin = bptr;
								goto quit_loop_2;
							case 1:
								bptr += inc_len;
								memcpy(tptr, hunk_begin, bptr - hunk_begin);
								tptr += (bptr - hunk_begin);
								hunk_begin = bptr;
								state = 0;
								break;
							default:
								bptr += inc_len;
								break;
						}
						break;
				}
				inc_len = (bptr < limit ? (*bptr == '\0' ? 1 : php_mblen(bptr, limit - bptr)): 0);
			}

		quit_loop_2:
			/* look up for a delimiter */
			for (;;) {
				switch (inc_len) {
					case 0:
						goto quit_loop_3;

					case -2:
					case -1:
						inc_len = 1;
						php_mb_reset();
						ZEND_FALLTHROUGH;
					case 1:
						if (*bptr == delimiter) {
							goto quit_loop_3;
						}
						break;
					default:
						break;
				}
				bptr += inc_len;
				inc_len = (bptr < limit ? (*bptr == '\0' ? 1 : php_mblen(bptr, limit - bptr)): 0);
			}

		quit_loop_3:
			memcpy(tptr, hunk_begin, bptr - hunk_begin);
			tptr += (bptr - hunk_begin);
			bptr += inc_len;
			comp_end = tptr;
		} else {
			/* 2B. Handle non-enclosure field */

			hunk_begin = bptr;

			for (;;) {
				switch (inc_len) {
					case 0:
						goto quit_loop_4;
					case -2:
					case -1:
						inc_len = 1;
						php_mb_reset();
						ZEND_FALLTHROUGH;
					case 1:
						if (*bptr == delimiter) {
							goto quit_loop_4;
						}
						break;
					default:
						break;
				}
				bptr += inc_len;
				inc_len = (bptr < limit ? (*bptr == '\0' ? 1 : php_mblen(bptr, limit - bptr)): 0);
			}
		quit_loop_4:
			memcpy(tptr, hunk_begin, bptr - hunk_begin);
			tptr += (bptr - hunk_begin);

			comp_end = (char *)php_fgetcsv_lookup_trailing_spaces(temp, tptr - temp);
			if (*bptr == delimiter) {
				bptr++;
			}
		}

		/* 3. Now pass our field back to php */
		*comp_end = '\0';

		zval z_tmp;
		ZVAL_STRINGL(&z_tmp, temp, comp_end - temp);
		zend_hash_next_index_insert(values, &z_tmp);
	} while (inc_len > 0);

	efree(temp);
	if (csv_stream) {
		efree(buf);
	}

	return values;
}
