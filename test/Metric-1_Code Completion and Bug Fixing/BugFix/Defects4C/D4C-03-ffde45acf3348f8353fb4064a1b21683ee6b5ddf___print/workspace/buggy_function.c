static int
bgp_attr_print(netdissect_options *ndo,
               u_int atype, const u_char *pptr, u_int len)
{
	int i;
	uint16_t af;
	uint8_t safi, snpa, nhlen;
        union { /* copy buffer for bandwidth values */
            float f;
            uint32_t i;
        } bw;
	int advance;
	u_int tlen;
	const u_char *tptr;
	char buf[MAXHOSTNAMELEN + 100];
        int  as_size;

        tptr = pptr;
        tlen=len;

	switch (atype) {
	case BGPTYPE_ORIGIN:
		if (len != 1)
			ND_PRINT((ndo, "invalid len"));
		else {
			ND_TCHECK(*tptr);
			ND_PRINT((ndo, "%s", tok2str(bgp_origin_values,
						"Unknown Origin Typecode",
						tptr[0])));
		}
		break;

        /*
         * Process AS4 byte path and AS2 byte path attributes here.
         */
	case BGPTYPE_AS4_PATH:
	case BGPTYPE_AS_PATH:
		if (len % 2) {
			ND_PRINT((ndo, "invalid len"));
			break;
		}
                if (!len) {
			ND_PRINT((ndo, "empty"));
			break;
                }

                /*
                 * BGP updates exchanged between New speakers that support 4
                 * byte AS, ASs are always encoded in 4 bytes. There is no
                 * definitive way to find this, just by the packet's
                 * contents. So, check for packet's TLV's sanity assuming
                 * 2 bytes first, and it does not pass, assume that ASs are
                 * encoded in 4 bytes format and move on.
                 */
                as_size = bgp_attr_get_as_size(ndo, atype, pptr, len);

		while (tptr < pptr + len) {
			ND_TCHECK(tptr[0]);
                        ND_PRINT((ndo, "%s", tok2str(bgp_as_path_segment_open_values,
						"?", tptr[0])));
			ND_TCHECK(tptr[1]);
                        for (i = 0; i < tptr[1] * as_size; i += as_size) {
                            ND_TCHECK2(tptr[2 + i], as_size);
			    ND_PRINT((ndo, "%s ",
				as_printf(ndo, astostr, sizeof(astostr),
				as_size == 2 ?
				EXTRACT_16BITS(&tptr[2 + i]) :
				EXTRACT_32BITS(&tptr[2 + i]))));
                        }
			ND_TCHECK(tptr[0]);
                        ND_PRINT((ndo, "%s", tok2str(bgp_as_path_segment_close_values,
						"?", tptr[0])));
                        ND_TCHECK(tptr[1]);
                        tptr += 2 + tptr[1] * as_size;
		}
		break;
	case BGPTYPE_NEXT_HOP:
		if (len != 4)
			ND_PRINT((ndo, "invalid len"));
		else {
			ND_TCHECK2(tptr[0], 4);
			ND_PRINT((ndo, "%s", ipaddr_string(ndo, tptr)));
		}
		break;
	case BGPTYPE_MULTI_EXIT_DISC:
	case BGPTYPE_LOCAL_PREF:
		if (len != 4)
			ND_PRINT((ndo, "invalid len"));
		else {
			ND_TCHECK2(tptr[0], 4);
			ND_PRINT((ndo, "%u", EXTRACT_32BITS(tptr)));
		}
		break;
	case BGPTYPE_ATOMIC_AGGREGATE:
		if (len != 0)
			ND_PRINT((ndo, "invalid len"));
		break;
        case BGPTYPE_AGGREGATOR:

                /*
                 * Depending on the AS encoded is of 2 bytes or of 4 bytes,
                 * the length of this PA can be either 6 bytes or 8 bytes.
                 */
                if (len != 6 && len != 8) {
                    ND_PRINT((ndo, "invalid len"));
                    break;
                }
                ND_TCHECK2(tptr[0], len);
                if (len == 6) {
		    ND_PRINT((ndo, " AS #%s, origin %s",
			as_printf(ndo, astostr, sizeof(astostr), EXTRACT_16BITS(tptr)),
			ipaddr_string(ndo, tptr + 2)));
                } else {
		    ND_PRINT((ndo, " AS #%s, origin %s",
			as_printf(ndo, astostr, sizeof(astostr),
			EXTRACT_32BITS(tptr)), ipaddr_string(ndo, tptr + 4)));
                }
                break;
	case BGPTYPE_AGGREGATOR4:
		if (len != 8) {
			ND_PRINT((ndo, "invalid len"));
			break;
		}
		ND_TCHECK2(tptr[0], 8);
		ND_PRINT((ndo, " AS #%s, origin %s",
	   	    as_printf(ndo, astostr, sizeof(astostr), EXTRACT_32BITS(tptr)),
		    ipaddr_string(ndo, tptr + 4)));
		break;
	case BGPTYPE_COMMUNITIES:
		if (len % 4) {
			ND_PRINT((ndo, "invalid len"));
			break;
		}
		while (tlen>0) {
			uint32_t comm;
			ND_TCHECK2(tptr[0], 4);
			comm = EXTRACT_32BITS(tptr);
			switch (comm) {
			case BGP_COMMUNITY_NO_EXPORT:
				ND_PRINT((ndo, " NO_EXPORT"));
				break;
			case BGP_COMMUNITY_NO_ADVERT:
				ND_PRINT((ndo, " NO_ADVERTISE"));
				break;
			case BGP_COMMUNITY_NO_EXPORT_SUBCONFED:
				ND_PRINT((ndo, " NO_EXPORT_SUBCONFED"));
				break;
			default:
				ND_PRINT((ndo, "%u:%u%s",
                                       (comm >> 16) & 0xffff,
                                       comm & 0xffff,
                                       (tlen>4) ? ", " : ""));
				break;
			}
                        tlen -=4;
                        tptr +=4;
		}
		break;
        case BGPTYPE_ORIGINATOR_ID:
		if (len != 4) {
			ND_PRINT((ndo, "invalid len"));
			break;
		}
		ND_TCHECK2(tptr[0], 4);
                ND_PRINT((ndo, "%s",ipaddr_string(ndo, tptr)));
                break;
        case BGPTYPE_CLUSTER_LIST:
		if (len % 4) {
			ND_PRINT((ndo, "invalid len"));
			break;
		}
                while (tlen>0) {
			ND_TCHECK2(tptr[0], 4);
                        ND_PRINT((ndo, "%s%s",
                               ipaddr_string(ndo, tptr),
                                (tlen>4) ? ", " : ""));
                        tlen -=4;
                        tptr +=4;
                }
                break;
	case BGPTYPE_MP_REACH_NLRI:
		ND_TCHECK2(tptr[0], 3);
		af = EXTRACT_16BITS(tptr);
		safi = tptr[2];

                ND_PRINT((ndo, "\n\t    AFI: %s (%u), %sSAFI: %s (%u)",
                       tok2str(af_values, "Unknown AFI", af),
                       af,
                       (safi>128) ? "vendor specific " : "", /* 128 is meanwhile wellknown */
                       tok2str(bgp_safi_values, "Unknown SAFI", safi),
                       safi));

                switch(af<<8 | safi) {
                case (AFNUM_INET<<8 | SAFNUM_UNICAST):
                case (AFNUM_INET<<8 | SAFNUM_MULTICAST):
                case (AFNUM_INET<<8 | SAFNUM_UNIMULTICAST):
                case (AFNUM_INET<<8 | SAFNUM_LABUNICAST):
                case (AFNUM_INET<<8 | SAFNUM_RT_ROUTING_INFO):
                case (AFNUM_INET<<8 | SAFNUM_VPNUNICAST):
                case (AFNUM_INET<<8 | SAFNUM_VPNMULTICAST):
                case (AFNUM_INET<<8 | SAFNUM_VPNUNIMULTICAST):
                case (AFNUM_INET<<8 | SAFNUM_MULTICAST_VPN):
		case (AFNUM_INET<<8 | SAFNUM_MDT):
                case (AFNUM_INET6<<8 | SAFNUM_UNICAST):
                case (AFNUM_INET6<<8 | SAFNUM_MULTICAST):
                case (AFNUM_INET6<<8 | SAFNUM_UNIMULTICAST):
                case (AFNUM_INET6<<8 | SAFNUM_LABUNICAST):
                case (AFNUM_INET6<<8 | SAFNUM_VPNUNICAST):
                case (AFNUM_INET6<<8 | SAFNUM_VPNMULTICAST):
                case (AFNUM_INET6<<8 | SAFNUM_VPNUNIMULTICAST):
                case (AFNUM_NSAP<<8 | SAFNUM_UNICAST):
                case (AFNUM_NSAP<<8 | SAFNUM_MULTICAST):
                case (AFNUM_NSAP<<8 | SAFNUM_UNIMULTICAST):
                case (AFNUM_NSAP<<8 | SAFNUM_VPNUNICAST):
                case (AFNUM_NSAP<<8 | SAFNUM_VPNMULTICAST):
                case (AFNUM_NSAP<<8 | SAFNUM_VPNUNIMULTICAST):
                case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNICAST):
                case (AFNUM_L2VPN<<8 | SAFNUM_VPNMULTICAST):
                case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNIMULTICAST):
                case (AFNUM_VPLS<<8 | SAFNUM_VPLS):
                    break;
                default:
                    ND_TCHECK2(tptr[0], tlen);
                    ND_PRINT((ndo, "\n\t    no AFI %u / SAFI %u decoder", af, safi));
                    if (ndo->ndo_vflag <= 1)
                        print_unknown_data(ndo, tptr, "\n\t    ", tlen);
                    goto done;
                    break;
                }

                tptr +=3;

		ND_TCHECK(tptr[0]);
		nhlen = tptr[0];
                tlen = nhlen;
                tptr++;

		if (tlen) {
                    int nnh = 0;
                    ND_PRINT((ndo, "\n\t    nexthop: "));
                    while (tlen > 0) {
                        if ( nnh++ > 0 ) {
                            ND_PRINT((ndo,  ", " ));
                        }
                        switch(af<<8 | safi) {
                        case (AFNUM_INET<<8 | SAFNUM_UNICAST):
                        case (AFNUM_INET<<8 | SAFNUM_MULTICAST):
                        case (AFNUM_INET<<8 | SAFNUM_UNIMULTICAST):
                        case (AFNUM_INET<<8 | SAFNUM_LABUNICAST):
                        case (AFNUM_INET<<8 | SAFNUM_RT_ROUTING_INFO):
                        case (AFNUM_INET<<8 | SAFNUM_MULTICAST_VPN):
                        case (AFNUM_INET<<8 | SAFNUM_MDT):
			    if (tlen < (int)sizeof(struct in_addr)) {
                                ND_PRINT((ndo, "invalid len"));
                                tlen = 0;
                            } else {
                                ND_TCHECK2(tptr[0], sizeof(struct in_addr));
                                ND_PRINT((ndo, "%s",ipaddr_string(ndo, tptr)));
                                tlen -= sizeof(struct in_addr);
                                tptr += sizeof(struct in_addr);
                            }
                            break;
                        case (AFNUM_INET<<8 | SAFNUM_VPNUNICAST):
                        case (AFNUM_INET<<8 | SAFNUM_VPNMULTICAST):
                        case (AFNUM_INET<<8 | SAFNUM_VPNUNIMULTICAST):
                            if (tlen < (int)(sizeof(struct in_addr)+BGP_VPN_RD_LEN)) {
                                ND_PRINT((ndo, "invalid len"));
                                tlen = 0;
                            } else {
                                ND_TCHECK2(tptr[0], sizeof(struct in_addr)+BGP_VPN_RD_LEN);
                                ND_PRINT((ndo, "RD: %s, %s",
                                       bgp_vpn_rd_print(ndo, tptr),
                                       ipaddr_string(ndo, tptr+BGP_VPN_RD_LEN)));
                                tlen -= (sizeof(struct in_addr)+BGP_VPN_RD_LEN);
                                tptr += (sizeof(struct in_addr)+BGP_VPN_RD_LEN);
                            }
                            break;
                        case (AFNUM_INET6<<8 | SAFNUM_UNICAST):
                        case (AFNUM_INET6<<8 | SAFNUM_MULTICAST):
                        case (AFNUM_INET6<<8 | SAFNUM_UNIMULTICAST):
                        case (AFNUM_INET6<<8 | SAFNUM_LABUNICAST):
                            if (tlen < (int)sizeof(struct in6_addr)) {
                                ND_PRINT((ndo, "invalid len"));
                                tlen = 0;
                            } else {
                                ND_TCHECK2(tptr[0], sizeof(struct in6_addr));
                                ND_PRINT((ndo, "%s", ip6addr_string(ndo, tptr)));
                                tlen -= sizeof(struct in6_addr);
                                tptr += sizeof(struct in6_addr);
                            }
                            break;
                        case (AFNUM_INET6<<8 | SAFNUM_VPNUNICAST):
                        case (AFNUM_INET6<<8 | SAFNUM_VPNMULTICAST):
                        case (AFNUM_INET6<<8 | SAFNUM_VPNUNIMULTICAST):
                            if (tlen < (int)(sizeof(struct in6_addr)+BGP_VPN_RD_LEN)) {
                                ND_PRINT((ndo, "invalid len"));
                                tlen = 0;
                            } else {
                                ND_TCHECK2(tptr[0], sizeof(struct in6_addr)+BGP_VPN_RD_LEN);
                                ND_PRINT((ndo, "RD: %s, %s",
                                       bgp_vpn_rd_print(ndo, tptr),
                                       ip6addr_string(ndo, tptr+BGP_VPN_RD_LEN)));
                                tlen -= (sizeof(struct in6_addr)+BGP_VPN_RD_LEN);
                                tptr += (sizeof(struct in6_addr)+BGP_VPN_RD_LEN);
                            }
                            break;
                        case (AFNUM_VPLS<<8 | SAFNUM_VPLS):
                        case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNICAST):
                        case (AFNUM_L2VPN<<8 | SAFNUM_VPNMULTICAST):
                        case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNIMULTICAST):
                            if (tlen < (int)sizeof(struct in_addr)) {
                                ND_PRINT((ndo, "invalid len"));
                                tlen = 0;
                            } else {
                                ND_TCHECK2(tptr[0], sizeof(struct in_addr));
                                ND_PRINT((ndo, "%s", ipaddr_string(ndo, tptr)));
                                tlen -= (sizeof(struct in_addr));
                                tptr += (sizeof(struct in_addr));
                            }
                            break;
                        case (AFNUM_NSAP<<8 | SAFNUM_UNICAST):
                        case (AFNUM_NSAP<<8 | SAFNUM_MULTICAST):
                        case (AFNUM_NSAP<<8 | SAFNUM_UNIMULTICAST):
                            ND_TCHECK2(tptr[0], tlen);
                            ND_PRINT((ndo, "%s", isonsap_string(ndo, tptr, tlen)));
                            tptr += tlen;
                            tlen = 0;
                            break;

                        case (AFNUM_NSAP<<8 | SAFNUM_VPNUNICAST):
                        case (AFNUM_NSAP<<8 | SAFNUM_VPNMULTICAST):
                        case (AFNUM_NSAP<<8 | SAFNUM_VPNUNIMULTICAST):
                            if (tlen < BGP_VPN_RD_LEN+1) {
                                ND_PRINT((ndo, "invalid len"));
                                tlen = 0;
                            } else {
                                ND_TCHECK2(tptr[0], tlen);
                                ND_PRINT((ndo, "RD: %s, %s",
                                       bgp_vpn_rd_print(ndo, tptr),
                                       isonsap_string(ndo, tptr+BGP_VPN_RD_LEN,tlen-BGP_VPN_RD_LEN)));
                                /* rfc986 mapped IPv4 address ? */
                                if (EXTRACT_32BITS(tptr+BGP_VPN_RD_LEN) ==  0x47000601)
                                    ND_PRINT((ndo, " = %s", ipaddr_string(ndo, tptr+BGP_VPN_RD_LEN+4)));
                                /* rfc1888 mapped IPv6 address ? */
                                else if (EXTRACT_24BITS(tptr+BGP_VPN_RD_LEN) ==  0x350000)
                                    ND_PRINT((ndo, " = %s", ip6addr_string(ndo, tptr+BGP_VPN_RD_LEN+3)));
                                tptr += tlen;
                                tlen = 0;
                            }
                            break;
                        default:
                            ND_TCHECK2(tptr[0], tlen);
                            ND_PRINT((ndo, "no AFI %u/SAFI %u decoder", af, safi));
                            if (ndo->ndo_vflag <= 1)
                                print_unknown_data(ndo, tptr, "\n\t    ", tlen);
                            tptr += tlen;
                            tlen = 0;
                            goto done;
                            break;
                        }
                    }
		}
		ND_PRINT((ndo, ", nh-length: %u", nhlen));
		tptr += tlen;

		ND_TCHECK(tptr[0]);
		snpa = tptr[0];
		tptr++;

		if (snpa) {
			ND_PRINT((ndo, "\n\t    %u SNPA", snpa));
			for (/*nothing*/; snpa > 0; snpa--) {
				ND_TCHECK(tptr[0]);
				ND_PRINT((ndo, "\n\t      %d bytes", tptr[0]));
				tptr += tptr[0] + 1;
			}
		} else {
			ND_PRINT((ndo, ", no SNPA"));
                }

		while (len - (tptr - pptr) > 0) {
                    switch (af<<8 | safi) {
                    case (AFNUM_INET<<8 | SAFNUM_UNICAST):
                    case (AFNUM_INET<<8 | SAFNUM_MULTICAST):
                    case (AFNUM_INET<<8 | SAFNUM_UNIMULTICAST):
                        advance = decode_prefix4(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET<<8 | SAFNUM_LABUNICAST):
                        advance = decode_labeled_prefix4(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_INET<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_INET<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_prefix4(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET<<8 | SAFNUM_RT_ROUTING_INFO):
                        advance = decode_rt_routing_info(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET<<8 | SAFNUM_MULTICAST_VPN): /* fall through */
                    case (AFNUM_INET6<<8 | SAFNUM_MULTICAST_VPN):
                        advance = decode_multicast_vpn(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;

		    case (AFNUM_INET<<8 | SAFNUM_MDT):
		      advance = decode_mdt_vpn_nlri(ndo, tptr, buf, sizeof(buf));
		      if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
		       break;
                    case (AFNUM_INET6<<8 | SAFNUM_UNICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_MULTICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_UNIMULTICAST):
                        advance = decode_prefix6(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET6<<8 | SAFNUM_LABUNICAST):
                        advance = decode_labeled_prefix6(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET6<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_prefix6(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_VPLS<<8 | SAFNUM_VPLS):
                    case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_L2VPN<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_l2(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_NSAP<<8 | SAFNUM_UNICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_MULTICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_UNIMULTICAST):
                        advance = decode_clnp_prefix(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_NSAP<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_clnp_prefix(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    default:
                        ND_TCHECK2(*tptr,tlen);
                        ND_PRINT((ndo, "\n\t    no AFI %u / SAFI %u decoder", af, safi));
                        if (ndo->ndo_vflag <= 1)
                            print_unknown_data(ndo, tptr, "\n\t    ", tlen);
                        advance = 0;
                        tptr = pptr + len;
                        break;
                    }
                    if (advance < 0)
                        break;
                    tptr += advance;
		}
        done:
		break;

	case BGPTYPE_MP_UNREACH_NLRI:
		ND_TCHECK2(tptr[0], BGP_MP_NLRI_MINSIZE);
		af = EXTRACT_16BITS(tptr);
		safi = tptr[2];

                ND_PRINT((ndo, "\n\t    AFI: %s (%u), %sSAFI: %s (%u)",
                       tok2str(af_values, "Unknown AFI", af),
                       af,
                       (safi>128) ? "vendor specific " : "", /* 128 is meanwhile wellknown */
                       tok2str(bgp_safi_values, "Unknown SAFI", safi),
                       safi));

                if (len == BGP_MP_NLRI_MINSIZE)
                    ND_PRINT((ndo, "\n\t      End-of-Rib Marker (empty NLRI)"));

		tptr += 3;

		while (len - (tptr - pptr) > 0) {
                    switch (af<<8 | safi) {
                    case (AFNUM_INET<<8 | SAFNUM_UNICAST):
                    case (AFNUM_INET<<8 | SAFNUM_MULTICAST):
                    case (AFNUM_INET<<8 | SAFNUM_UNIMULTICAST):
                        advance = decode_prefix4(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET<<8 | SAFNUM_LABUNICAST):
                        advance = decode_labeled_prefix4(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_INET<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_INET<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_prefix4(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET6<<8 | SAFNUM_UNICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_MULTICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_UNIMULTICAST):
                        advance = decode_prefix6(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET6<<8 | SAFNUM_LABUNICAST):
                        advance = decode_labeled_prefix6(ndo, tptr, len, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else if (advance == -3)
                            break; /* bytes left, but not enough */
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_INET6<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_INET6<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_prefix6(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_VPLS<<8 | SAFNUM_VPLS):
                    case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_L2VPN<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_L2VPN<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_l2(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_NSAP<<8 | SAFNUM_UNICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_MULTICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_UNIMULTICAST):
                        advance = decode_clnp_prefix(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    case (AFNUM_NSAP<<8 | SAFNUM_VPNUNICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_VPNMULTICAST):
                    case (AFNUM_NSAP<<8 | SAFNUM_VPNUNIMULTICAST):
                        advance = decode_labeled_vpn_clnp_prefix(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
		    case (AFNUM_INET<<8 | SAFNUM_MDT):
		      advance = decode_mdt_vpn_nlri(ndo, tptr, buf, sizeof(buf));
		      if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
		       break;
                    case (AFNUM_INET<<8 | SAFNUM_MULTICAST_VPN): /* fall through */
                    case (AFNUM_INET6<<8 | SAFNUM_MULTICAST_VPN):
                        advance = decode_multicast_vpn(ndo, tptr, buf, sizeof(buf));
                        if (advance == -1)
                            ND_PRINT((ndo, "\n\t    (illegal prefix length)"));
                        else if (advance == -2)
                            goto trunc;
                        else
                            ND_PRINT((ndo, "\n\t      %s", buf));
                        break;
                    default:
                        ND_TCHECK2(*(tptr-3),tlen);
                        ND_PRINT((ndo, "no AFI %u / SAFI %u decoder", af, safi));
                        if (ndo->ndo_vflag <= 1)
                            print_unknown_data(ndo, tptr-3, "\n\t    ", tlen);
                        advance = 0;
                        tptr = pptr + len;
                        break;
                    }
                    if (advance < 0)
                        break;
                    tptr += advance;
		}
		break;
        case BGPTYPE_EXTD_COMMUNITIES:
		if (len % 8) {
			ND_PRINT((ndo, "invalid len"));
			break;
		}
                while (tlen>0) {
                    uint16_t extd_comm;

                    ND_TCHECK2(tptr[0], 2);
                    extd_comm=EXTRACT_16BITS(tptr);

		    ND_PRINT((ndo, "\n\t    %s (0x%04x), Flags [%s]",
			   tok2str(bgp_extd_comm_subtype_values,
				      "unknown extd community typecode",
				      extd_comm),
			   extd_comm,
			   bittok2str(bgp_extd_comm_flag_values, "none", extd_comm)));

                    ND_TCHECK2(*(tptr+2), 6);
                    switch(extd_comm) {
                    case BGP_EXT_COM_RT_0:
                    case BGP_EXT_COM_RO_0:
                    case BGP_EXT_COM_L2VPN_RT_0:
                        ND_PRINT((ndo, ": %u:%u (= %s)",
                               EXTRACT_16BITS(tptr+2),
                               EXTRACT_32BITS(tptr+4),
                               ipaddr_string(ndo, tptr+4)));
                        break;
                    case BGP_EXT_COM_RT_1:
                    case BGP_EXT_COM_RO_1:
                    case BGP_EXT_COM_L2VPN_RT_1:
                    case BGP_EXT_COM_VRF_RT_IMP:
                        ND_PRINT((ndo, ": %s:%u",
                               ipaddr_string(ndo, tptr+2),
                               EXTRACT_16BITS(tptr+6)));
                        break;
                    case BGP_EXT_COM_RT_2:
                    case BGP_EXT_COM_RO_2:
			ND_PRINT((ndo, ": %s:%u",
			    as_printf(ndo, astostr, sizeof(astostr),
			    EXTRACT_32BITS(tptr+2)), EXTRACT_16BITS(tptr+6)));
			break;
                    case BGP_EXT_COM_LINKBAND:
		        bw.i = EXTRACT_32BITS(tptr+2);
                        ND_PRINT((ndo, ": bandwidth: %.3f Mbps",
                               bw.f*8/1000000));
                        break;
                    case BGP_EXT_COM_VPN_ORIGIN:
                    case BGP_EXT_COM_VPN_ORIGIN2:
                    case BGP_EXT_COM_VPN_ORIGIN3:
                    case BGP_EXT_COM_VPN_ORIGIN4:
                    case BGP_EXT_COM_OSPF_RID:
                    case BGP_EXT_COM_OSPF_RID2:
                        ND_PRINT((ndo, "%s", ipaddr_string(ndo, tptr+2)));
                        break;
                    case BGP_EXT_COM_OSPF_RTYPE:
                    case BGP_EXT_COM_OSPF_RTYPE2:
                        ND_PRINT((ndo, ": area:%s, router-type:%s, metric-type:%s%s",
                               ipaddr_string(ndo, tptr+2),
                               tok2str(bgp_extd_comm_ospf_rtype_values,
					  "unknown (0x%02x)",
					  *(tptr+6)),
                               (*(tptr+7) &  BGP_OSPF_RTYPE_METRIC_TYPE) ? "E2" : "",
                               ((*(tptr+6) == BGP_OSPF_RTYPE_EXT) || (*(tptr+6) == BGP_OSPF_RTYPE_NSSA)) ? "E1" : ""));
                        break;
                    case BGP_EXT_COM_L2INFO:
                        ND_PRINT((ndo, ": %s Control Flags [0x%02x]:MTU %u",
                               tok2str(l2vpn_encaps_values,
					  "unknown encaps",
					  *(tptr+2)),
                                       *(tptr+3),
                               EXTRACT_16BITS(tptr+4)));
                        break;
                    case BGP_EXT_COM_SOURCE_AS:
                        ND_PRINT((ndo, ": AS %u", EXTRACT_16BITS(tptr+2)));
                        break;
                    default:
                        ND_TCHECK2(*tptr,8);
                        print_unknown_data(ndo, tptr, "\n\t      ", 8);
                        break;
                    }
                    tlen -=8;
                    tptr +=8;
                }
                break;

        case BGPTYPE_PMSI_TUNNEL:
        {
                uint8_t tunnel_type, flags;

                tunnel_type = *(tptr+1);
                flags = *tptr;
                tlen = len;

                ND_TCHECK2(tptr[0], 5);
                ND_PRINT((ndo, "\n\t    Tunnel-type %s (%u), Flags [%s], MPLS Label %u",
                       tok2str(bgp_pmsi_tunnel_values, "Unknown", tunnel_type),
                       tunnel_type,
                       bittok2str(bgp_pmsi_flag_values, "none", flags),
                       EXTRACT_24BITS(tptr+2)>>4));

                tptr +=5;
                tlen -= 5;

                switch (tunnel_type) {
                case BGP_PMSI_TUNNEL_PIM_SM: /* fall through */
                case BGP_PMSI_TUNNEL_PIM_BIDIR:
                    ND_TCHECK2(tptr[0], 8);
                    ND_PRINT((ndo, "\n\t      Sender %s, P-Group %s",
                           ipaddr_string(ndo, tptr),
                           ipaddr_string(ndo, tptr+4)));
                    break;

                case BGP_PMSI_TUNNEL_PIM_SSM:
                    ND_TCHECK2(tptr[0], 8);
                    ND_PRINT((ndo, "\n\t      Root-Node %s, P-Group %s",
                           ipaddr_string(ndo, tptr),
                           ipaddr_string(ndo, tptr+4)));
                    break;
                case BGP_PMSI_TUNNEL_INGRESS:
                    ND_TCHECK2(tptr[0], 4);
                    ND_PRINT((ndo, "\n\t      Tunnel-Endpoint %s",
                           ipaddr_string(ndo, tptr)));
                    break;
                case BGP_PMSI_TUNNEL_LDP_P2MP: /* fall through */
                case BGP_PMSI_TUNNEL_LDP_MP2MP:
                    ND_TCHECK2(tptr[0], 8);
                    ND_PRINT((ndo, "\n\t      Root-Node %s, LSP-ID 0x%08x",
                           ipaddr_string(ndo, tptr),
                           EXTRACT_32BITS(tptr+4)));
                    break;
                case BGP_PMSI_TUNNEL_RSVP_P2MP:
                    ND_TCHECK2(tptr[0], 8);
                    ND_PRINT((ndo, "\n\t      Extended-Tunnel-ID %s, P2MP-ID 0x%08x",
                           ipaddr_string(ndo, tptr),
                           EXTRACT_32BITS(tptr+4)));
                    break;
                default:
                    if (ndo->ndo_vflag <= 1) {
                        print_unknown_data(ndo, tptr, "\n\t      ", tlen);
                    }
                }
                break;
        }
	case BGPTYPE_AIGP:
	{
		uint8_t type;
		uint16_t length;

		ND_TCHECK2(tptr[0], 3);

		tlen = len;

		while (tlen >= 3) {

		    type = *tptr;
		    length = EXTRACT_16BITS(tptr+1);

		    ND_PRINT((ndo, "\n\t    %s TLV (%u), length %u",
			      tok2str(bgp_aigp_values, "Unknown", type),
			      type, length));

		    /*
		     * Check if we can read the TLV data.
		     */
		    ND_TCHECK2(tptr[3], length - 3);

		    switch (type) {

		    case BGP_AIGP_TLV:
		        ND_TCHECK2(tptr[3], 8);
			ND_PRINT((ndo, ", metric %" PRIu64,
				  EXTRACT_64BITS(tptr+3)));
			break;

		    default:
			if (ndo->ndo_vflag <= 1) {
			    print_unknown_data(ndo, tptr+3,"\n\t      ", length-3);
			}
		    }

		    tptr += length;
		    tlen -= length;
		}
		break;
	}
        case BGPTYPE_ATTR_SET:
                ND_TCHECK2(tptr[0], 4);
                if (len < 4)
                	goto trunc;
		ND_PRINT((ndo, "\n\t    Origin AS: %s",
		    as_printf(ndo, astostr, sizeof(astostr), EXTRACT_32BITS(tptr))));
		tptr+=4;
                len -=4;

                while (len) {
                    u_int aflags, alenlen, alen;

                    ND_TCHECK2(tptr[0], 2);
                    if (len < 2)
                        goto trunc;
                    aflags = *tptr;
                    atype = *(tptr + 1);
                    tptr += 2;
                    len -= 2;
                    alenlen = bgp_attr_lenlen(aflags, tptr);
                    ND_TCHECK2(tptr[0], alenlen);
                    if (len < alenlen)
                        goto trunc;
                    alen = bgp_attr_len(aflags, tptr);
                    tptr += alenlen;
                    len -= alenlen;

                    ND_PRINT((ndo, "\n\t      %s (%u), length: %u",
                           tok2str(bgp_attr_values,
                                      "Unknown Attribute", atype),
                           atype,
                           alen));

                    if (aflags) {
                        ND_PRINT((ndo, ", Flags [%s%s%s%s",
                               aflags & 0x80 ? "O" : "",
                               aflags & 0x40 ? "T" : "",
                               aflags & 0x20 ? "P" : "",
                               aflags & 0x10 ? "E" : ""));
                        if (aflags & 0xf)
                            ND_PRINT((ndo, "+%x", aflags & 0xf));
                        ND_PRINT((ndo, "]: "));
                    }
                    /* FIXME check for recursion */
                    if (!bgp_attr_print(ndo, atype, tptr, alen))
                        return 0;
                    tptr += alen;
                    len -= alen;
		}
                break;

	case BGPTYPE_LARGE_COMMUNITY:
		if (len == 0 || len % 12) {
			ND_PRINT((ndo, "invalid len"));
			break;
		}
		ND_PRINT((ndo, "\n\t    "));
		while (len > 0) {
			ND_TCHECK2(*tptr, 12);
			ND_PRINT((ndo, "%u:%u:%u%s",
				 EXTRACT_32BITS(tptr),
				 EXTRACT_32BITS(tptr + 4),
				 EXTRACT_32BITS(tptr + 8),
				 (len > 12) ? ", " : ""));
                        tptr += 12;
                        len -= 12;
		}
		break;
	default:
	    ND_TCHECK2(*pptr,len);
            ND_PRINT((ndo, "\n\t    no Attribute %u decoder", atype)); /* we have no decoder for the attribute */
            if (ndo->ndo_vflag <= 1)
                print_unknown_data(ndo, pptr, "\n\t    ", len);
            break;
	}
        if (ndo->ndo_vflag > 1 && len) { /* omit zero length attributes*/
            ND_TCHECK2(*pptr,len);
            print_unknown_data(ndo, pptr, "\n\t    ", len);
        }
        return 1;

trunc:
        return 0;
}
