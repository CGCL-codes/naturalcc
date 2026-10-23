static int
parse_elements(netdissect_options *ndo,
               struct mgmt_body_t *pbody, const u_char *p, int offset,
               u_int length)
{
	u_int elementlen;
	struct ssid_t ssid;
	struct challenge_t challenge;
	struct rates_t rates;
	struct ds_t ds;
	struct cf_t cf;
	struct tim_t tim;

	/*
	 * We haven't seen any elements yet.
	 */
	pbody->challenge_present = 0;
	pbody->ssid_present = 0;
	pbody->rates_present = 0;
	pbody->ds_present = 0;
	pbody->cf_present = 0;
	pbody->tim_present = 0;

	while (length != 0) {
		/* Make sure we at least have the element ID and length. */
		if (!ND_TTEST2(*(p + offset), 2))
			return 0;
		if (length < 2)
			return 0;
		elementlen = *(p + offset + 1);

		/* Make sure we have the entire element. */
		if (!ND_TTEST2(*(p + offset + 2), elementlen))
			return 0;
		if (length < elementlen + 2)
			return 0;

		switch (*(p + offset)) {
		case E_SSID:
			memcpy(&ssid, p + offset, 2);
			offset += 2;
			length -= 2;
			if (ssid.length != 0) {
				if (ssid.length > sizeof(ssid.ssid) - 1)
					return 0;
				if (!ND_TTEST2(*(p + offset), ssid.length))
					return 0;
				if (length < ssid.length)
					return 0;
				memcpy(&ssid.ssid, p + offset, ssid.length);
				offset += ssid.length;
				length -= ssid.length;
			}
			ssid.ssid[ssid.length] = '\0';
			/*
			 * Present and not truncated.
			 *
			 * If we haven't already seen an SSID IE,
			 * copy this one, otherwise ignore this one,
			 * so we later report the first one we saw.
			 */
			if (!pbody->ssid_present) {
				pbody->ssid = ssid;
				pbody->ssid_present = 1;
			}
			break;
		case E_CHALLENGE:
			memcpy(&challenge, p + offset, 2);
			offset += 2;
			length -= 2;
			if (challenge.length != 0) {
				if (challenge.length >
				    sizeof(challenge.text) - 1)
					return 0;
				if (!ND_TTEST2(*(p + offset), challenge.length))
					return 0;
				if (length < challenge.length)
					return 0;
				memcpy(&challenge.text, p + offset,
				    challenge.length);
				offset += challenge.length;
				length -= challenge.length;
			}
			challenge.text[challenge.length] = '\0';
			/*
			 * Present and not truncated.
			 *
			 * If we haven't already seen a challenge IE,
			 * copy this one, otherwise ignore this one,
			 * so we later report the first one we saw.
			 */
			if (!pbody->challenge_present) {
				pbody->challenge = challenge;
				pbody->challenge_present = 1;
			}
			break;
		case E_RATES:
			memcpy(&rates, p + offset, 2);
			offset += 2;
			length -= 2;
			if (rates.length != 0) {
				if (rates.length > sizeof rates.rate)
					return 0;
				if (!ND_TTEST2(*(p + offset), rates.length))
					return 0;
				if (length < rates.length)
					return 0;
				memcpy(&rates.rate, p + offset, rates.length);
				offset += rates.length;
				length -= rates.length;
			}
			/*
			 * Present and not truncated.
			 *
			 * If we haven't already seen a rates IE,
			 * copy this one if it's not zero-length,
			 * otherwise ignore this one, so we later
			 * report the first one we saw.
			 *
			 * We ignore zero-length rates IEs as some
			 * devices seem to put a zero-length rates
			 * IE, followed by an SSID IE, followed by
			 * a non-zero-length rates IE into frames,
			 * even though IEEE Std 802.11-2007 doesn't
			 * seem to indicate that a zero-length rates
			 * IE is valid.
			 */
			if (!pbody->rates_present && rates.length != 0) {
				pbody->rates = rates;
				pbody->rates_present = 1;
			}
			break;
		case E_DS:
			memcpy(&ds, p + offset, 2);
			offset += 2;
			length -= 2;
			if (ds.length != 1) {
				offset += ds.length;
				length -= ds.length;
				break;
			}
			ds.channel = *(p + offset);
			offset += 1;
			length -= 1;
			/*
			 * Present and not truncated.
			 *
			 * If we haven't already seen a DS IE,
			 * copy this one, otherwise ignore this one,
			 * so we later report the first one we saw.
			 */
			if (!pbody->ds_present) {
				pbody->ds = ds;
				pbody->ds_present = 1;
			}
			break;
		case E_CF:
			memcpy(&cf, p + offset, 2);
			offset += 2;
			length -= 2;
			if (cf.length != 6) {
				offset += cf.length;
				length -= cf.length;
				break;
			}
			memcpy(&cf.count, p + offset, 6);
			offset += 6;
			length -= 6;
			/*
			 * Present and not truncated.
			 *
			 * If we haven't already seen a CF IE,
			 * copy this one, otherwise ignore this one,
			 * so we later report the first one we saw.
			 */
			if (!pbody->cf_present) {
				pbody->cf = cf;
				pbody->cf_present = 1;
			}
			break;
		case E_TIM:
			memcpy(&tim, p + offset, 2);
			offset += 2;
			length -= 2;
			if (tim.length <= 3) {
				offset += tim.length;
				length -= tim.length;
				break;
			}
			if (tim.length - 3 > (int)sizeof tim.bitmap)
				return 0;
			memcpy(&tim.count, p + offset, 3);
			offset += 3;
			length -= 3;

			memcpy(tim.bitmap, p + offset + 3, tim.length - 3);
			offset += tim.length - 3;
			length -= tim.length - 3;
			/*
			 * Present and not truncated.
			 *
			 * If we haven't already seen a TIM IE,
			 * copy this one, otherwise ignore this one,
			 * so we later report the first one we saw.
			 */
			if (!pbody->tim_present) {
				pbody->tim = tim;
				pbody->tim_present = 1;
			}
			break;
		default:
#if 0
			ND_PRINT((ndo, "(1) unhandled element_id (%d)  ",
			    *(p + offset)));
#endif
			offset += 2 + elementlen;
			length -= 2 + elementlen;
			break;
		}
	}

	/* No problems found. */
	return 1;
}
