void processCertificateElements(struct ndpi_detection_module_struct *detectionmodulestruct,
				struct ndpi_flow_struct *flow,
				u_int16_t p_offset, u_int16_t certificate_len) {
  // Initialize variables required for processing certificate elements.
  struct ndpi_packet_struct *packet = &detectionmodulestruct->packet;
  u_int16_t num_found = 0, i;
  char buffer[64] = { '\0' }, rdnSeqBuf[2048];
  u_int rdn_len = 0;

  rdnSeqBuf[0] = '\0';

#ifdef DEBUG_TLS
  printf("[TLS] %s() [offset: %u][certificate_len: %u]\n", __FUNCTION__, p_offset, certificate_len);
#endif

  /* Check after handshake protocol header (5 bytes) and message header (4 bytes) */
  for(i = p_offset; i < certificate_len - 2; i++) {
    /*
      See https://www.ibm.com/support/knowledgecenter/SSFKSJ_7.5.0/com.ibm.mq.sec.doc/q009860_.htm
      for X.509 certificate labels
    */
    if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x04) && (packet->payload[i+2] == 0x03)) {
      /* Common Name */
      int rc = extractRDNSequence(packet, i, buffer, sizeof(buffer), rdnSeqBuf, &rdn_len, sizeof(rdnSeqBuf), "CN");
      if(rc == -1) break;

#ifdef DEBUG_TLS
      printf("[TLS] %s() [%s][%s: %s]\n", __FUNCTION__, (num_found == 0) ? "Subject" : "Issuer", "Common Name", buffer);
#endif
    } else if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x04) && (packet->payload[i+2] == 0x06)) {
      /* Country */
      int rc = extractRDNSequence(packet, i, buffer, sizeof(buffer), rdnSeqBuf, &rdn_len, sizeof(rdnSeqBuf), "C");
      if(rc == -1) break;

#ifdef DEBUG_TLS
      printf("[TLS] %s() [%s][%s: %s]\n", __FUNCTION__, (num_found == 0) ? "Subject" : "Issuer", "Country", buffer);
#endif
    } else if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x04) && (packet->payload[i+2] == 0x07)) {
      /* Locality */
      int rc = extractRDNSequence(packet, i, buffer, sizeof(buffer), rdnSeqBuf, &rdn_len, sizeof(rdnSeqBuf), "L");
      if(rc == -1) break;

#ifdef DEBUG_TLS
      printf("[TLS] %s() [%s][%s: %s]\n", __FUNCTION__, (num_found == 0) ? "Subject" : "Issuer", "Locality", buffer);
#endif
    } else if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x04) && (packet->payload[i+2] == 0x08)) {
      /* State or Province */
      int rc = extractRDNSequence(packet, i, buffer, sizeof(buffer), rdnSeqBuf, &rdn_len, sizeof(rdnSeqBuf), "ST");
      if(rc == -1) break;

#ifdef DEBUG_TLS
      printf("[TLS] %s() [%s][%s: %s]\n", __FUNCTION__, (num_found == 0) ? "Subject" : "Issuer", "State or Province", buffer);
#endif
    } else if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x04) && (packet->payload[i+2] == 0x0a)) {
      /* Organization Name */
      int rc = extractRDNSequence(packet, i, buffer, sizeof(buffer), rdnSeqBuf, &rdn_len, sizeof(rdnSeqBuf), "O");
      if(rc == -1) break;

#ifdef DEBUG_TLS
      printf("[TLS] %s() [%s][%s: %s]\n", __FUNCTION__, (num_found == 0) ? "Subject" : "Issuer", "Organization Name", buffer);
#endif

    } else if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x04) && (packet->payload[i+2] == 0x0b)) {
      /* Organization Unit */
      int rc = extractRDNSequence(packet, i, buffer, sizeof(buffer), rdnSeqBuf, &rdn_len, sizeof(rdnSeqBuf), "OU");
      if(rc == -1) break;

#ifdef DEBUG_TLS
      printf("[TLS] %s() [%s][%s: %s]\n", __FUNCTION__, (num_found == 0) ? "Subject" : "Issuer", "Organization Unit", buffer);
#endif
    } else if((packet->payload[i] == 0x30) && (packet->payload[i+1] == 0x1e) && (packet->payload[i+2] == 0x17)) {
      /* Certificate Validity */
      u_int offset = i+4;

      if(num_found == 0) {
	num_found++;

#ifdef DEBUG_TLS
	printf("[TLS] %s() IssuerDN [%s]\n", __FUNCTION__, rdnSeqBuf);
#endif

	if(rdn_len && (flow->protos.tls_quic.issuerDN == NULL)) {
	  flow->protos.tls_quic.issuerDN = ndpi_strdup(rdnSeqBuf);
	  if(ndpi_normalize_printable_string(rdnSeqBuf, rdn_len) == 0) {
	    char str[64];

	    snprintf(str, sizeof(str), "Invalid issuerDN %s", flow->protos.tls_quic.issuerDN);
	    ndpi_set_risk(detectionmodulestruct, flow, NDPI_INVALID_CHARACTERS, str);
	  }
	}

	rdn_len = 0; /* Reset buffer */
      }

      if(i + 3 < certificate_len &&
	 (offset+packet->payload[i+3]) < packet->payload_packet_len) {
	char utcDate[32];
        u_int8_t len = packet->payload[i+3];

#ifdef DEBUG_TLS
	u_int j;

	printf("[CERTIFICATE] notBefore [len: %u][", len);
	for(j=0; j<len; j++) printf("%c", packet->payload[i+4+j]);
	printf("]\n");
#endif

	if(len < (sizeof(utcDate)-1)) {
	  struct tm utc;
	  utc.tm_isdst = -1; /* Not set by strptime */

	  strncpy(utcDate, (const char*)&packet->payload[i+4], len);
	  utcDate[len] = '\0';

	  /* 141021000000Z */
	  if(strptime(utcDate, "%y%m%d%H%M%SZ", &utc) != NULL) {
	    flow->protos.tls_quic.notBefore = timegm(&utc);
#ifdef DEBUG_TLS
	    printf("[CERTIFICATE] notBefore %u [%s]\n",
		   flow->protos.tls_quic.notBefore, utcDate);
#endif
	  }
	}

	offset += len;

	if((offset+1) < packet->payload_packet_len) {
	  len = packet->payload[offset+1];

	  offset += 2;

	  if((offset+len) < packet->payload_packet_len) {
	    u_int32_t time_sec = packet->current_time_ms / 1000;
#ifdef DEBUG_TLS
	    u_int j;

	    printf("[CERTIFICATE] notAfter [len: %u][", len);
	    for(j=0; j<len; j++) printf("%c", packet->payload[offset+j]);
	    printf("]\n");
#endif

	    if(len < (sizeof(utcDate)-1)) {
	      struct tm utc;
	      utc.tm_isdst = -1; /* Not set by strptime */

	      strncpy(utcDate, (const char*)&packet->payload[offset], len);
	      utcDate[len] = '\0';

	      /* 141021000000Z */
	      if(strptime(utcDate, "%y%m%d%H%M%SZ", &utc) != NULL) {
		flow->protos.tls_quic.notAfter = timegm(&utc);
#ifdef DEBUG_TLS
		printf("[CERTIFICATE] notAfter %u [%s]\n",
		       flow->protos.tls_quic.notAfter, utcDate);
#endif
	      }
	    }

	    if(flow->protos.tls_quic.notBefore > TLS_LIMIT_DATE)
	      if((flow->protos.tls_quic.notAfter-flow->protos.tls_quic.notBefore) > TLS_THRESHOLD) {
		char str[64];
		
		snprintf(str, sizeof(str), "TLS Cert lasts %u days",
			  (flow->protos.tls_quic.notAfter-flow->protos.tls_quic.notBefore) / 86400);

		ndpi_set_risk(detectionmodulestruct, flow, NDPI_TLS_CERT_VALIDITY_TOO_LONG, str); /* Certificate validity longer than 13 months */
	      }
	    
	    if((time_sec < flow->protos.tls_quic.notBefore) || (time_sec > flow->protos.tls_quic.notAfter)) {
	      char str[96], b[32], e[32];
	      struct tm result;	      
	      time_t theTime;

	      theTime = flow->protos.tls_quic.notBefore;
	      strftime(b, sizeof(b), "%d/%b/%Y %H:%M:%S", ndpi_gmtime_r(&theTime, &result));

	      theTime = flow->protos.tls_quic.notAfter;
	      strftime(e, sizeof(e), "%d/%b/%Y %H:%M:%S", ndpi_gmtime_r(&theTime, &result));
	      	      
	      snprintf(str, sizeof(str), "%s - %s", b, e);
	      ndpi_set_risk(detectionmodulestruct, flow, NDPI_TLS_CERTIFICATE_EXPIRED, str); /* Certificate expired */
	    } else if((time_sec > flow->protos.tls_quic.notBefore)
		      && (time_sec > (flow->protos.tls_quic.notAfter - (detectionmodulestruct->tls_certificate_expire_in_x_days * 86400)))) {
	      char str[96], b[32], e[32];
	      struct tm result;	      
	      time_t theTime;

	      theTime = flow->protos.tls_quic.notBefore;
	      strftime(b, sizeof(b), "%d/%b/%Y %H:%M:%S", ndpi_gmtime_r(&theTime, &result));

	      theTime = flow->protos.tls_quic.notAfter;
	      strftime(e, sizeof(e), "%d/%b/%Y %H:%M:%S", ndpi_gmtime_r(&theTime, &result));

	      snprintf(str, sizeof(str), "%s - %s", b, e);
	      ndpi_set_risk(detectionmodulestruct, flow, NDPI_TLS_CERTIFICATE_ABOUT_TO_EXPIRE, str); /* Certificate almost expired */
	    }
	  }
	}
      }
    } else if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x1d) && (packet->payload[i+2] == 0x11)) {
      /* Organization OID: 2.5.29.17 (subjectAltName) */
      u_int8_t matched_name = 0;

      /* If the client hello was not observed or the requested name was missing, there is no need to trigger an alert */
      if(flow->host_server_name[0] == '\0')
	matched_name = 1;

#ifdef DEBUG_TLS
      printf("******* [TLS] Found subjectAltName\n");
#endif

      i += 3 /* skip the initial patten 55 1D 11 */;

      /* skip the first type, 0x04 == BIT STRING, and jump to it's length */
      if(i < packet->payload_packet_len && packet->payload[i] == 0x04) i++; else i += 4; /* 4 bytes, with the last byte set to 04 */

      if(i < packet->payload_packet_len) {
	i += (packet->payload[i] & 0x80) ? (packet->payload[i] & 0x7F) : 0; /* skip BIT STRING length */
	if(i < packet->payload_packet_len) {
	  i += 2; /* skip the second type, 0x30 == SEQUENCE, and jump to it's length */
	  if(i < packet->payload_packet_len) {
	    i += (packet->payload[i] & 0x80) ? (packet->payload[i] & 0x7F) : 0; /* skip SEQUENCE length */
	    i++;

	    while(i < packet->payload_packet_len) {
	      u_int8_t general_name_type = packet->payload[i];

	      if((general_name_type == 0x81)    /* rfc822Name */
		 || (general_name_type == 0x82) /* dNSName    */
		 || (general_name_type == 0x87) /* ipAddress  */
		 )
		{
		  if((i < (packet->payload_packet_len - 1))
		     && ((i + packet->payload[i + 1] + 2) < packet->payload_packet_len)) {
		    u_int8_t len = packet->payload[i + 1];
		    char dNSName[256];
		    u_int16_t dNSName_len;

		    i += 2;

		    /* The check "len > sizeof(dNSName) - 1" will be always false. If we add it,
		       the compiler is smart enough to detect it and throws a warning */
		    if((len == 0 /* Looks something went wrong */)
		       || ((i+len) > packet->payload_packet_len))
		      break;

		    if(general_name_type == 0x87) {
		      if(len == 4 /* IPv4 */) {
			ndpi_snprintf(dNSName, sizeof(dNSName), "%u.%u.%u.%u",
				      packet->payload[i] & 0xFF,
				      packet->payload[i+1] & 0xFF,
				      packet->payload[i+2] & 0xFF,
				      packet->payload[i+3] & 0xFF);
		      } else if(len == 16 /* IPv6 */) {
			struct in6_addr addr = *(struct in6_addr *)&packet->payload[i];
			inet_ntop(AF_INET6, &addr, dNSName, sizeof(dNSName));
		      } else {
			/* Is that possibile? Better safe than sorry */
			dNSName[0] = '\0';
		      }
		    } else {
		      strncpy(dNSName, (const char*)&packet->payload[i], len);
		      dNSName[len] = '\0';
		    }

		    dNSName_len = strlen(dNSName);
		    cleanupServerName(dNSName, dNSName_len);

#if DEBUG_TLS
		    printf("[TLS] dNSName %s [%s][len: %u][leftover: %d]\n", dNSName,
			   flow->host_server_name, len,
			   packet->payload_packet_len-i-len);
#endif

		    /*
		      We cannot use ndpi_is_valid_hostname() as we can have wildcards
		      here that will create false positives
		    */
		    if(ndpi_normalize_printable_string(dNSName, dNSName_len) == 0) {
		      ndpi_set_risk(detectionmodulestruct, flow, NDPI_INVALID_CHARACTERS, dNSName);

		      /* This looks like an attack */
		      ndpi_set_risk(detectionmodulestruct, flow, NDPI_POSSIBLE_EXPLOIT, NULL);
		    }

		    if(matched_name == 0) {
#if DEBUG_TLS
		      printf("[TLS] Trying to match '%s' with '%s'\n",
			     flow->host_server_name,
			     dNSName);
#endif

		      if(flow->host_server_name[0] == '\0') {
			matched_name = 1;	/* No SNI */
		      } else if(dNSName[0] == '*') {
			char * label = strstr(flow->host_server_name, &dNSName[1]);

			if(label != NULL) {
			  char * first_dot = strchr(flow->host_server_name, '.');

			  if((first_dot == NULL) || (first_dot <= label)) {
			    matched_name = 1;
			  }
			}
		      }
		      else if(strcmp(flow->host_server_name, dNSName) == 0) {
			matched_name = 1;
		      }
		    }

		    if(flow->protos.tls_quic.server_names == NULL)
		      flow->protos.tls_quic.server_names = ndpi_strdup(dNSName),
			flow->protos.tls_quic.server_names_len = strlen(dNSName);
		    else {
		      u_int16_t newstr_len = flow->protos.tls_quic.server_names_len + dNSName_len + 1;
		      char *newstr = (char*)ndpi_realloc(flow->protos.tls_quic.server_names,
							 flow->protos.tls_quic.server_names_len+1, newstr_len+1);

		      if(newstr) {
			flow->protos.tls_quic.server_names = newstr;
			flow->protos.tls_quic.server_names[flow->protos.tls_quic.server_names_len] = ',';
			strncpy(&flow->protos.tls_quic.server_names[flow->protos.tls_quic.server_names_len+1],
				dNSName, dNSName_len+1);
			flow->protos.tls_quic.server_names[newstr_len] = '\0';
			flow->protos.tls_quic.server_names_len = newstr_len;
		      }
		    }

		    if(!flow->protos.tls_quic.subprotocol_detected)
		      if(ndpi_match_hostname_protocol(detectionmodulestruct, flow, __get_master(detectionmodulestruct, flow), dNSName, dNSName_len)) {
			flow->protos.tls_quic.subprotocol_detected = 1;
		        ndpi_unset_risk(detectionmodulestruct, flow, NDPI_NUMERIC_IP_HOST);
		      }

		    i += len;
		  } else {
#if DEBUG_TLS
		    printf("[TLS] Leftover %u bytes", packet->payload_packet_len - i);
#endif
		    ndpi_set_risk(detectionmodulestruct, flow, NDPI_TLS_SUSPICIOUS_EXTENSION, NULL);
		    break;
		  }
		} else {
		break;
	      }
	    } /* while */

	    if(!matched_name) {
	      char str[128];

	      snprintf(str, sizeof(str), "%s vs %s", flow->host_server_name, flow->protos.tls_quic.server_names);
	      ndpi_set_risk(detectionmodulestruct, flow, NDPI_TLS_CERTIFICATE_MISMATCH, str); /* Certificate mismatch */
	    }
	  }
	}
      }
    }
  } /* for */

  if(rdn_len && (flow->protos.tls_quic.subjectDN == NULL)) {
    flow->protos.tls_quic.subjectDN = ndpi_strdup(rdnSeqBuf);

    if(flow->detected_protocol_stack[1] == NDPI_PROTOCOL_UNKNOWN) {
      /* No idea what is happening behind the scenes: let's check the certificate */
      u_int32_t val;
      int rc = ndpi_match_string_value(detectionmodulestruct->tls_cert_subject_automa.ac_automa,
				       rdnSeqBuf, strlen(rdnSeqBuf), &val);

      if(rc == 0) {
	/* Match found */
	u_int16_t proto_id = (u_int16_t)val;
	ndpi_protocol ret = { __get_master(detectionmodulestruct, flow), proto_id, NDPI_PROTOCOL_UNKNOWN /* unused */, NDPI_PROTOCOL_CATEGORY_UNSPECIFIED, NULL};

	ndpi_set_detected_protocol(detectionmodulestruct, flow, proto_id, __get_master(detectionmodulestruct, flow), NDPI_CONFIDENCE_DPI);
	flow->category = ndpi_get_proto_category(detectionmodulestruct, ret);
	ndpi_check_subprotocol_risk(detectionmodulestruct, flow, proto_id);
	ndpi_unset_risk(detectionmodulestruct, flow, NDPI_NUMERIC_IP_HOST);

	if(detectionmodulestruct->tls_cert_cache) {
	  u_int32_t key = make_tls_cert_key(packet, 0 /* from the server */);

	  ndpi_lru_add_to_cache(detectionmodulestruct->tls_cert_cache, key, proto_id, ndpi_get_current_time(flow));
	}
      }
    }
  }

  if(flow->protos.tls_quic.subjectDN && flow->protos.tls_quic.issuerDN
     && (!strcmp(flow->protos.tls_quic.subjectDN, flow->protos.tls_quic.issuerDN))) {
    /* Last resort: we check if this is a trusted issuerDN */
    if(ndpi_check_issuerdn_risk_exception(detectionmodulestruct, flow->protos.tls_quic.issuerDN))
      return; /* This is a trusted DN */
    
    ndpi_set_risk(detectionmodulestruct, flow, NDPI_TLS_SELFSIGNED_CERTIFICATE, flow->protos.tls_quic.subjectDN);
  }
  
#if DEBUG_TLS
  printf("[TLS] %s() SubjectDN [%s]\n", __FUNCTION__, rdnSeqBuf);
#endif
}
