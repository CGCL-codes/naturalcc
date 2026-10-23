void
pgm_print(netdissect_options *ndo,
          register const u_char *bp, register u_int length,
          register const u_char *bp2)
{
	register const struct pgm_header *pgm;
	register const struct ip *ip;
	register char ch;
	uint16_t sport, dport;
	u_int nla_afnum;
	char nla_buf[INET6_ADDRSTRLEN];
	register const struct ip6_hdr *ip6;
	uint8_t opt_type, opt_len;
	uint32_t seq, opts_len, len, offset;

	pgm = (const struct pgm_header *)bp;
	ip = (const struct ip *)bp2;
	if (IP_V(ip) == 6)
		ip6 = (const struct ip6_hdr *)bp2;
	else
		ip6 = NULL;
	ch = '\0';
	if (!ND_TTEST(pgm->pgm_dport)) {
		if (ip6) {
			ND_PRINT((ndo, "%s > %s: [|pgm]",
				ip6addr_string(ndo, &ip6->ip6_src),
				ip6addr_string(ndo, &ip6->ip6_dst)));
			return;
		} else {
			ND_PRINT((ndo, "%s > %s: [|pgm]",
				ipaddr_string(ndo, &ip->ip_src),
				ipaddr_string(ndo, &ip->ip_dst)));
			return;
		}
	}

	sport = EXTRACT_16BITS(&pgm->pgm_sport);
	dport = EXTRACT_16BITS(&pgm->pgm_dport);

	if (ip6) {
		if (ip6->ip6_nxt == IPPROTO_PGM) {
			ND_PRINT((ndo, "%s.%s > %s.%s: ",
				ip6addr_string(ndo, &ip6->ip6_src),
				tcpport_string(ndo, sport),
				ip6addr_string(ndo, &ip6->ip6_dst),
				tcpport_string(ndo, dport)));
		} else {
			ND_PRINT((ndo, "%s > %s: ",
				tcpport_string(ndo, sport), tcpport_string(ndo, dport)));
		}
	} else {
		if (ip->ip_p == IPPROTO_PGM) {
			ND_PRINT((ndo, "%s.%s > %s.%s: ",
				ipaddr_string(ndo, &ip->ip_src),
				tcpport_string(ndo, sport),
				ipaddr_string(ndo, &ip->ip_dst),
				tcpport_string(ndo, dport)));
		} else {
			ND_PRINT((ndo, "%s > %s: ",
				tcpport_string(ndo, sport), tcpport_string(ndo, dport)));
		}
	}

	ND_TCHECK(*pgm);

        ND_PRINT((ndo, "PGM, length %u", EXTRACT_16BITS(&pgm->pgm_length)));

        if (!ndo->ndo_vflag)
            return;

	ND_PRINT((ndo, " 0x%02x%02x%02x%02x%02x%02x ",
		     pgm->pgm_gsid[0],
                     pgm->pgm_gsid[1],
                     pgm->pgm_gsid[2],
		     pgm->pgm_gsid[3],
                     pgm->pgm_gsid[4],
                     pgm->pgm_gsid[5]));
	switch (pgm->pgm_type) {
	case PGM_SPM: {
	    const struct pgm_spm *spm;

	    spm = (const struct pgm_spm *)(pgm + 1);
	    ND_TCHECK(*spm);
	    bp = (const u_char *) (spm + 1);

	    switch (EXTRACT_16BITS(&spm->pgms_nla_afi)) {
	    case AFNUM_INET:
		ND_TCHECK2(*bp, sizeof(struct in_addr));
		addrtostr(bp, nla_buf, sizeof(nla_buf));
		bp += sizeof(struct in_addr);
		break;
	    case AFNUM_INET6:
		ND_TCHECK2(*bp, sizeof(struct in6_addr));
		addrtostr6(bp, nla_buf, sizeof(nla_buf));
		bp += sizeof(struct in6_addr);
		break;
	    default:
		goto trunc;
		break;
	    }

	    ND_PRINT((ndo, "SPM seq %u trail %u lead %u nla %s",
			 EXTRACT_32BITS(&spm->pgms_seq),
                         EXTRACT_32BITS(&spm->pgms_trailseq),
			 EXTRACT_32BITS(&spm->pgms_leadseq),
                         nla_buf));
	    break;
	}

	case PGM_POLL: {
	    const struct pgm_poll *poll_msg;

	    poll_msg = (const struct pgm_poll *)(pgm + 1);
	    ND_TCHECK(*poll_msg);
	    ND_PRINT((ndo, "POLL seq %u round %u",
			 EXTRACT_32BITS(&poll_msg->pgmp_seq),
                         EXTRACT_16BITS(&poll_msg->pgmp_round)));
	    bp = (const u_char *) (poll_msg + 1);
	    break;
	}
	case PGM_POLR: {
	    const struct pgm_polr *polr;
	    uint32_t ivl, rnd, mask;

	    polr = (const struct pgm_polr *)(pgm + 1);
	    ND_TCHECK(*polr);
	    bp = (const u_char *) (polr + 1);

	    switch (EXTRACT_16BITS(&polr->pgmp_nla_afi)) {
	    case AFNUM_INET:
		ND_TCHECK2(*bp, sizeof(struct in_addr));
		addrtostr(bp, nla_buf, sizeof(nla_buf));
		bp += sizeof(struct in_addr);
		break;
	    case AFNUM_INET6:
		ND_TCHECK2(*bp, sizeof(struct in6_addr));
		addrtostr6(bp, nla_buf, sizeof(nla_buf));
		bp += sizeof(struct in6_addr);
		break;
	    default:
		goto trunc;
		break;
	    }

	    ND_TCHECK2(*bp, sizeof(uint32_t));
	    ivl = EXTRACT_32BITS(bp);
	    bp += sizeof(uint32_t);

	    ND_TCHECK2(*bp, sizeof(uint32_t));
	    rnd = EXTRACT_32BITS(bp);
	    bp += sizeof(uint32_t);

	    ND_TCHECK2(*bp, sizeof(uint32_t));
	    mask = EXTRACT_32BITS(bp);
	    bp += sizeof(uint32_t);

	    ND_PRINT((ndo, "POLR seq %u round %u nla %s ivl %u rnd 0x%08x "
			 "mask 0x%08x", EXTRACT_32BITS(&polr->pgmp_seq),
			 EXTRACT_16BITS(&polr->pgmp_round), nla_buf, ivl, rnd, mask));
	    break;
	}
	case PGM_ODATA: {
	    const struct pgm_data *odata;

	    odata = (const struct pgm_data *)(pgm + 1);
	    ND_TCHECK(*odata);
	    ND_PRINT((ndo, "ODATA trail %u seq %u",
			 EXTRACT_32BITS(&odata->pgmd_trailseq),
			 EXTRACT_32BITS(&odata->pgmd_seq)));
	    bp = (const u_char *) (odata + 1);
	    break;
	}

	case PGM_RDATA: {
	    const struct pgm_data *rdata;

	    rdata = (const struct pgm_data *)(pgm + 1);
	    ND_TCHECK(*rdata);
	    ND_PRINT((ndo, "RDATA trail %u seq %u",
			 EXTRACT_32BITS(&rdata->pgmd_trailseq),
			 EXTRACT_32BITS(&rdata->pgmd_seq)));
	    bp = (const u_char *) (rdata + 1);
	    break;
	}

	case PGM_NAK:
	case PGM_NULLNAK:
	case PGM_NCF: {
	    const struct pgm_nak *nak;
	    char source_buf[INET6_ADDRSTRLEN], group_buf[INET6_ADDRSTRLEN];

	    nak = (const struct pgm_nak *)(pgm + 1);
	    ND_TCHECK(*nak);
	    bp = (const u_char *) (nak + 1);

	    /*
	     * Skip past the source, saving info along the way
	     * and stopping if we don't have enough.
	     */
	    switch (EXTRACT_16BITS(&nak->pgmn_source_afi)) {
	    case AFNUM_INET:
		ND_TCHECK2(*bp, sizeof(struct in_addr));
		addrtostr(bp, source_buf, sizeof(source_buf));
		bp += sizeof(struct in_addr);
		break;
	    case AFNUM_INET6:
		ND_TCHECK2(*bp, sizeof(struct in6_addr));
		addrtostr6(bp, source_buf, sizeof(source_buf));
		bp += sizeof(struct in6_addr);
		break;
	    default:
		goto trunc;
		break;
	    }

	    /*
	     * Skip past the group, saving info along the way
	     * and stopping if we don't have enough.
	     */
	    bp += (2 * sizeof(uint16_t));
	    switch (EXTRACT_16BITS(bp)) {
	    case AFNUM_INET:
		ND_TCHECK2(*bp, sizeof(struct in_addr));
		addrtostr(bp, group_buf, sizeof(group_buf));
		bp += sizeof(struct in_addr);
		break;
	    case AFNUM_INET6:
		ND_TCHECK2(*bp, sizeof(struct in6_addr));
		addrtostr6(bp, group_buf, sizeof(group_buf));
		bp += sizeof(struct in6_addr);
		break;
	    default:
		goto trunc;
		break;
	    }

	    /*
	     * Options decoding can go here.
	     */
	    switch (pgm->pgm_type) {
		case PGM_NAK:
		    ND_PRINT((ndo, "NAK "));
		    break;
		case PGM_NULLNAK:
		    ND_PRINT((ndo, "NNAK "));
		    break;
		case PGM_NCF:
		    ND_PRINT((ndo, "NCF "));
		    break;
		default:
                    break;
	    }
	    ND_PRINT((ndo, "(%s -> %s), seq %u",
			 source_buf, group_buf, EXTRACT_32BITS(&nak->pgmn_seq)));
	    break;
	}

	case PGM_ACK: {
	    const struct pgm_ack *ack;

	    ack = (const struct pgm_ack *)(pgm + 1);
	    ND_TCHECK(*ack);
	    ND_PRINT((ndo, "ACK seq %u",
			 EXTRACT_32BITS(&ack->pgma_rx_max_seq)));
	    bp = (const u_char *) (ack + 1);
	    break;
	}

	case PGM_SPMR:
	    ND_PRINT((ndo, "SPMR"));
	    break;

	default:
	    ND_PRINT((ndo, "UNKNOWN type 0x%02x", pgm->pgm_type));
	    break;

	}
	if (pgm->pgm_options & PGM_OPT_BIT_PRESENT) {

	    /*
	     * make sure there's enough for the first option header
	     */
	    if (!ND_TTEST2(*bp, PGM_MIN_OPT_LEN)) {
		ND_PRINT((ndo, "[|OPT]"));
		return;
	    }

	    /*
	     * That option header MUST be an OPT_LENGTH option
	     * (see the first paragraph of section 9.1 in RFC 3208).
	     */
	    opt_type = *bp++;
	    if ((opt_type & PGM_OPT_MASK) != PGM_OPT_LENGTH) {
		ND_PRINT((ndo, "[First option bad, should be PGM_OPT_LENGTH, is %u]", opt_type & PGM_OPT_MASK));
		return;
	    }
	    opt_len = *bp++;
	    if (opt_len != 4) {
		ND_PRINT((ndo, "[Bad OPT_LENGTH option, length %u != 4]", opt_len));
		return;
	    }
	    opts_len = EXTRACT_16BITS(bp);
	    if (opts_len < 4) {
		ND_PRINT((ndo, "[Bad total option length %u < 4]", opts_len));
		return;
	    }
	    bp += sizeof(uint16_t);
	    ND_PRINT((ndo, " OPTS LEN %d", opts_len));
	    opts_len -= 4;

	    while (opts_len) {
		if (opts_len < PGM_MIN_OPT_LEN) {
		    ND_PRINT((ndo, "[Total option length leaves no room for final option]"));
		    return;
		}
		if (!ND_TTEST2(*bp, 2)) {
		    ND_PRINT((ndo, " [|OPT]"));
		    return;
		}
		opt_type = *bp++;
		opt_len = *bp++;
		if (opt_len < PGM_MIN_OPT_LEN) {
		    ND_PRINT((ndo, "[Bad option, length %u < %u]", opt_len,
		        PGM_MIN_OPT_LEN));
		    break;
		}
		if (opts_len < opt_len) {
		    ND_PRINT((ndo, "[Total option length leaves no room for final option]"));
		    return;
		}
		if (!ND_TTEST2(*bp, opt_len - 2)) {
		    ND_PRINT((ndo, " [|OPT]"));
		    return;
		}

		switch (opt_type & PGM_OPT_MASK) {
		case PGM_OPT_LENGTH:
#define PGM_OPT_LENGTH_LEN	(2+2)
		    if (opt_len != PGM_OPT_LENGTH_LEN) {
			ND_PRINT((ndo, "[Bad OPT_LENGTH option, length %u != %u]",
			    opt_len, PGM_OPT_LENGTH_LEN));
			return;
		    }
		    ND_PRINT((ndo, " OPTS LEN (extra?) %d", EXTRACT_16BITS(bp)));
		    bp += 2;
		    opts_len -= PGM_OPT_LENGTH_LEN;
		    break;

		case PGM_OPT_FRAGMENT:
#define PGM_OPT_FRAGMENT_LEN	(2+2+4+4+4)
		    if (opt_len != PGM_OPT_FRAGMENT_LEN) {
			ND_PRINT((ndo, "[Bad OPT_FRAGMENT option, length %u != %u]",
			    opt_len, PGM_OPT_FRAGMENT_LEN));
			return;
		    }
		    bp += 2;
		    seq = EXTRACT_32BITS(bp);
		    bp += 4;
		    offset = EXTRACT_32BITS(bp);
		    bp += 4;
		    len = EXTRACT_32BITS(bp);
		    bp += 4;
		    ND_PRINT((ndo, " FRAG seq %u off %u len %u", seq, offset, len));
		    opts_len -= PGM_OPT_FRAGMENT_LEN;
		    break;

		case PGM_OPT_NAK_LIST:
		    bp += 2;
		    opt_len -= 4;	/* option header */
		    ND_PRINT((ndo, " NAK LIST"));
		    while (opt_len) {
			if (opt_len < 4) {
			    ND_PRINT((ndo, "[Option length not a multiple of 4]"));
			    return;
			}
			ND_TCHECK2(*bp, 4);
			ND_PRINT((ndo, " %u", EXTRACT_32BITS(bp)));
			bp += 4;
			opt_len -= 4;
			opts_len -= 4;
		    }
		    break;

		case PGM_OPT_JOIN:
#define PGM_OPT_JOIN_LEN	(2+2+4)
		    if (opt_len != PGM_OPT_JOIN_LEN) {
			ND_PRINT((ndo, "[Bad OPT_JOIN option, length %u != %u]",
			    opt_len, PGM_OPT_JOIN_LEN));
			return;
		    }
		    bp += 2;
		    seq = EXTRACT_32BITS(bp);
		    bp += 4;
		    ND_PRINT((ndo, " JOIN %u", seq));
		    opts_len -= PGM_OPT_JOIN_LEN;
		    break;

		case PGM_OPT_NAK_BO_IVL:
#define PGM_OPT_NAK_BO_IVL_LEN	(2+2+4+4)
		    if (opt_len != PGM_OPT_NAK_BO_IVL_LEN) {
			ND_PRINT((ndo, "[Bad OPT_NAK_BO_IVL option, length %u != %u]",
			    opt_len, PGM_OPT_NAK_BO_IVL_LEN));
			return;
		    }
		    bp += 2;
		    offset = EXTRACT_32BITS(bp);
		    bp += 4;
		    seq = EXTRACT_32BITS(bp);
		    bp += 4;
		    ND_PRINT((ndo, " BACKOFF ivl %u ivlseq %u", offset, seq));
		    opts_len -= PGM_OPT_NAK_BO_IVL_LEN;
		    break;

		case PGM_OPT_NAK_BO_RNG:
#define PGM_OPT_NAK_BO_RNG_LEN	(2+2+4+4)
		    if (opt_len != PGM_OPT_NAK_BO_RNG_LEN) {
			ND_PRINT((ndo, "[Bad OPT_NAK_BO_RNG option, length %u != %u]",
			    opt_len, PGM_OPT_NAK_BO_RNG_LEN));
			return;
		    }
		    bp += 2;
		    offset = EXTRACT_32BITS(bp);
		    bp += 4;
		    seq = EXTRACT_32BITS(bp);
		    bp += 4;
		    ND_PRINT((ndo, " BACKOFF max %u min %u", offset, seq));
		    opts_len -= PGM_OPT_NAK_BO_RNG_LEN;
		    break;

		case PGM_OPT_REDIRECT:
#define PGM_OPT_REDIRECT_FIXED_LEN	(2+2+2+2)
		    if (opt_len < PGM_OPT_REDIRECT_FIXED_LEN) {
			ND_PRINT((ndo, "[Bad OPT_REDIRECT option, length %u < %u]",
			    opt_len, PGM_OPT_REDIRECT_FIXED_LEN));
			return;
		    }
		    bp += 2;
		    nla_afnum = EXTRACT_16BITS(bp);
		    bp += 2+2;
		    switch (nla_afnum) {
		    case AFNUM_INET:
			if (opt_len != PGM_OPT_REDIRECT_FIXED_LEN + sizeof(struct in_addr)) {
			    ND_PRINT((ndo, "[Bad OPT_REDIRECT option, length %u != %u + address size]",
			        opt_len, PGM_OPT_REDIRECT_FIXED_LEN));
			    return;
			}
			ND_TCHECK2(*bp, sizeof(struct in_addr));
			addrtostr(bp, nla_buf, sizeof(nla_buf));
			bp += sizeof(struct in_addr);
			opts_len -= PGM_OPT_REDIRECT_FIXED_LEN + sizeof(struct in_addr);
			break;
		    case AFNUM_INET6:
			if (opt_len != PGM_OPT_REDIRECT_FIXED_LEN + sizeof(struct in6_addr)) {
			    ND_PRINT((ndo, "[Bad OPT_REDIRECT option, length %u != %u + address size]",
			        PGM_OPT_REDIRECT_FIXED_LEN, opt_len));
			    return;
			}
			ND_TCHECK2(*bp, sizeof(struct in6_addr));
			addrtostr6(bp, nla_buf, sizeof(nla_buf));
			bp += sizeof(struct in6_addr);
			opts_len -= PGM_OPT_REDIRECT_FIXED_LEN + sizeof(struct in6_addr);
			break;
		    default:
			goto trunc;
			break;
		    }

		    ND_PRINT((ndo, " REDIRECT %s",  nla_buf));
		    break;

		case PGM_OPT_PARITY_PRM:
#define PGM_OPT_PARITY_PRM_LEN	(2+2+4)
		    if (opt_len != PGM_OPT_PARITY_PRM_LEN) {
			ND_PRINT((ndo, "[Bad OPT_PARITY_PRM option, length %u != %u]",
			    opt_len, PGM_OPT_PARITY_PRM_LEN));
			return;
		    }
		    bp += 2;
		    len = EXTRACT_32BITS(bp);
		    bp += 4;
		    ND_PRINT((ndo, " PARITY MAXTGS %u", len));
		    opts_len -= PGM_OPT_PARITY_PRM_LEN;
		    break;

		case PGM_OPT_PARITY_GRP:
#define PGM_OPT_PARITY_GRP_LEN	(2+2+4)
		    if (opt_len != PGM_OPT_PARITY_GRP_LEN) {
			ND_PRINT((ndo, "[Bad OPT_PARITY_GRP option, length %u != %u]",
			    opt_len, PGM_OPT_PARITY_GRP_LEN));
			return;
		    }
		    bp += 2;
		    seq = EXTRACT_32BITS(bp);
		    bp += 4;
		    ND_PRINT((ndo, " PARITY GROUP %u", seq));
		    opts_len -= PGM_OPT_PARITY_GRP_LEN;
		    break;

		case PGM_OPT_CURR_TGSIZE:
#define PGM_OPT_CURR_TGSIZE_LEN	(2+2+4)
		    if (opt_len != PGM_OPT_CURR_TGSIZE_LEN) {
			ND_PRINT((ndo, "[Bad OPT_CURR_TGSIZE option, length %u != %u]",
			    opt_len, PGM_OPT_CURR_TGSIZE_LEN));
			return;
		    }
		    bp += 2;
		    len = EXTRACT_32BITS(bp);
		    bp += 4;
		    ND_PRINT((ndo, " PARITY ATGS %u", len));
		    opts_len -= PGM_OPT_CURR_TGSIZE_LEN;
		    break;

		case PGM_OPT_NBR_UNREACH:
#define PGM_OPT_NBR_UNREACH_LEN	(2+2)
		    if (opt_len != PGM_OPT_NBR_UNREACH_LEN) {
			ND_PRINT((ndo, "[Bad OPT_NBR_UNREACH option, length %u != %u]",
			    opt_len, PGM_OPT_NBR_UNREACH_LEN));
			return;
		    }
		    bp += 2;
		    ND_PRINT((ndo, " NBR_UNREACH"));
		    opts_len -= PGM_OPT_NBR_UNREACH_LEN;
		    break;

		case PGM_OPT_PATH_NLA:
		    ND_PRINT((ndo, " PATH_NLA [%d]", opt_len));
		    bp += opt_len;
		    opts_len -= opt_len;
		    break;

		case PGM_OPT_SYN:
#define PGM_OPT_SYN_LEN	(2+2)
		    if (opt_len != PGM_OPT_SYN_LEN) {
			ND_PRINT((ndo, "[Bad OPT_SYN option, length %u != %u]",
			    opt_len, PGM_OPT_SYN_LEN));
			return;
		    }
		    bp += 2;
		    ND_PRINT((ndo, " SYN"));
		    opts_len -= PGM_OPT_SYN_LEN;
		    break;

		case PGM_OPT_FIN:
#define PGM_OPT_FIN_LEN	(2+2)
		    if (opt_len != PGM_OPT_FIN_LEN) {
			ND_PRINT((ndo, "[Bad OPT_FIN option, length %u != %u]",
			    opt_len, PGM_OPT_FIN_LEN));
			return;
		    }
		    bp += 2;
		    ND_PRINT((ndo, " FIN"));
		    opts_len -= PGM_OPT_FIN_LEN;
		    break;

		case PGM_OPT_RST:
#define PGM_OPT_RST_LEN	(2+2)
		    if (opt_len != PGM_OPT_RST_LEN) {
			ND_PRINT((ndo, "[Bad OPT_RST option, length %u != %u]",
			    opt_len, PGM_OPT_RST_LEN));
			return;
		    }
		    bp += 2;
		    ND_PRINT((ndo, " RST"));
		    opts_len -= PGM_OPT_RST_LEN;
		    break;

		case PGM_OPT_CR:
		    ND_PRINT((ndo, " CR"));
		    bp += opt_len;
		    opts_len -= opt_len;
		    break;

		case PGM_OPT_CRQST:
#define PGM_OPT_CRQST_LEN	(2+2)
		    if (opt_len != PGM_OPT_CRQST_LEN) {
			ND_PRINT((ndo, "[Bad OPT_CRQST option, length %u != %u]",
			    opt_len, PGM_OPT_CRQST_LEN));
			return;
		    }
		    bp += 2;
		    ND_PRINT((ndo, " CRQST"));
		    opts_len -= PGM_OPT_CRQST_LEN;
		    break;

		case PGM_OPT_PGMCC_DATA:
#define PGM_OPT_PGMCC_DATA_FIXED_LEN	(2+2+4+2+2)
		    if (opt_len < PGM_OPT_PGMCC_DATA_FIXED_LEN) {
			ND_PRINT((ndo, "[Bad OPT_PGMCC_DATA option, length %u < %u]",
			    opt_len, PGM_OPT_PGMCC_DATA_FIXED_LEN));
			return;
		    }
		    bp += 2;
		    offset = EXTRACT_32BITS(bp);
		    bp += 4;
		    nla_afnum = EXTRACT_16BITS(bp);
		    bp += 2+2;
		    switch (nla_afnum) {
		    case AFNUM_INET:
			if (opt_len != PGM_OPT_PGMCC_DATA_FIXED_LEN + sizeof(struct in_addr)) {
			    ND_PRINT((ndo, "[Bad OPT_PGMCC_DATA option, length %u != %u + address size]",
			        opt_len, PGM_OPT_PGMCC_DATA_FIXED_LEN));
			    return;
			}
			ND_TCHECK2(*bp, sizeof(struct in_addr));
			addrtostr(bp, nla_buf, sizeof(nla_buf));
			bp += sizeof(struct in_addr);
			opts_len -= PGM_OPT_PGMCC_DATA_FIXED_LEN + sizeof(struct in_addr);
			break;
		    case AFNUM_INET6:
			if (opt_len != PGM_OPT_PGMCC_DATA_FIXED_LEN + sizeof(struct in6_addr)) {
			    ND_PRINT((ndo, "[Bad OPT_PGMCC_DATA option, length %u != %u + address size]",
			        opt_len, PGM_OPT_PGMCC_DATA_FIXED_LEN));
			    return;
			}
			ND_TCHECK2(*bp, sizeof(struct in6_addr));
			addrtostr6(bp, nla_buf, sizeof(nla_buf));
			bp += sizeof(struct in6_addr);
			opts_len -= PGM_OPT_PGMCC_DATA_FIXED_LEN + sizeof(struct in6_addr);
			break;
		    default:
			goto trunc;
			break;
		    }

		    ND_PRINT((ndo, " PGMCC DATA %u %s", offset, nla_buf));
		    break;

		case PGM_OPT_PGMCC_FEEDBACK:
#define PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN	(2+2+4+2+2)
		    if (opt_len < PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN) {
			ND_PRINT((ndo, "[Bad PGM_OPT_PGMCC_FEEDBACK option, length %u < %u]",
			    opt_len, PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN));
			return;
		    }
		    bp += 2;
		    offset = EXTRACT_32BITS(bp);
		    bp += 4;
		    nla_afnum = EXTRACT_16BITS(bp);
		    bp += 2+2;
		    switch (nla_afnum) {
		    case AFNUM_INET:
			if (opt_len != PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN + sizeof(struct in_addr)) {
			    ND_PRINT((ndo, "[Bad OPT_PGMCC_FEEDBACK option, length %u != %u + address size]",
			        opt_len, PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN));
			    return;
			}
			ND_TCHECK2(*bp, sizeof(struct in_addr));
			addrtostr(bp, nla_buf, sizeof(nla_buf));
			bp += sizeof(struct in_addr);
			opts_len -= PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN + sizeof(struct in_addr);
			break;
		    case AFNUM_INET6:
			if (opt_len != PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN + sizeof(struct in6_addr)) {
			    ND_PRINT((ndo, "[Bad OPT_PGMCC_FEEDBACK option, length %u != %u + address size]",
			        opt_len, PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN));
			    return;
			}
			ND_TCHECK2(*bp, sizeof(struct in6_addr));
			addrtostr6(bp, nla_buf, sizeof(nla_buf));
			bp += sizeof(struct in6_addr);
			opts_len -= PGM_OPT_PGMCC_FEEDBACK_FIXED_LEN + sizeof(struct in6_addr);
			break;
		    default:
			goto trunc;
			break;
		    }

		    ND_PRINT((ndo, " PGMCC FEEDBACK %u %s", offset, nla_buf));
		    break;

		default:
		    ND_PRINT((ndo, " OPT_%02X [%d] ", opt_type, opt_len));
		    bp += opt_len;
		    opts_len -= opt_len;
		    break;
		}

		if (opt_type & PGM_OPT_END)
		    break;
	     }
	}

	ND_PRINT((ndo, " [%u]", length));
	if (ndo->ndo_packettype == PT_PGM_ZMTP1 &&
	    (pgm->pgm_type == PGM_ODATA || pgm->pgm_type == PGM_RDATA))
		zmtp1_print_datagram(ndo, bp, EXTRACT_16BITS(&pgm->pgm_length));

	return;

trunc:
	ND_PRINT((ndo, "[|pgm]"));
	if (ch != '\0')
		ND_PRINT((ndo, ">"));
}
