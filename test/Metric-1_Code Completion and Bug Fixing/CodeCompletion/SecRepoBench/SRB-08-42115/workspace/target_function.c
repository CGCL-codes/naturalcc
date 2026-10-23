static void processCertificateElements(struct ndpi_detection_module_struct *ndpi_struct,
				       struct ndpi_flow_struct *connection,
				       u_int16_t p_offset, u_int16_t certificate_len) {
  struct ndpi_packet_struct *packet = &ndpi_struct->packet;
  u_int16_t num_found = 0, i;
  char buffer[64] = { '\0' }, rdnSeqBuf[2048];
  u_int rdn_len = 0;

  rdnSeqBuf[0] = '\0';

#ifdef DEBUG_TLS
  printf("[TLS] %s() [offset: %u][certificate_len: %u]\n", __FUNCTION__, p_offset, certificate_len);
#endif

  /* Check after handshake protocol header (5 bytes) and message header (4 bytes) */
  for(i = p_offset; i < certificate_len; i++) {
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
      u_int8_t len = packet->payload[i+3];
      u_int offset = i+4;

      if(num_found == 0) {
	num_found++;

#ifdef DEBUG_TLS
	printf("[TLS] %s() IssuerDN [%s]\n", __FUNCTION__, rdnSeqBuf);
#endif

	if(rdn_len && (connection->protos.tls_quic.issuerDN == NULL)) {
	  connection->protos.tls_quic.issuerDN = ndpi_strdup(rdnSeqBuf);
	  if(ndpi_is_printable_string(rdnSeqBuf, rdn_len) == 0) {
	    ndpi_set_risk(ndpi_struct, connection, NDPI_INVALID_CHARACTERS);
	  }
	}

	rdn_len = 0; /* Reset buffer */
      }

      if((offset+len) < packet->payload_packet_len) {
	char utcDate[32];

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
	    connection->protos.tls_quic.notBefore = timegm(&utc);
#ifdef DEBUG_TLS
	    printf("[CERTIFICATE] notBefore %u [%s]\n",
		   connection->protos.tls_quic.notBefore, utcDate);
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
		connection->protos.tls_quic.notAfter = timegm(&utc);
#ifdef DEBUG_TLS
		printf("[CERTIFICATE] notAfter %u [%s]\n",
		       connection->protos.tls_quic.notAfter, utcDate);
#endif
	      }
	    }

	    if(connection->protos.tls_quic.notBefore > TLS_LIMIT_DATE)
	      if((connection->protos.tls_quic.notAfter-connection->protos.tls_quic.notBefore) > TLS_THRESHOLD)
		ndpi_set_risk(ndpi_struct, connection, NDPI_TLS_CERT_VALIDITY_TOO_LONG); /* Certificate validity longer than 13 months*/

	    if((time_sec < connection->protos.tls_quic.notBefore)
	       || (time_sec > connection->protos.tls_quic.notAfter))
	      ndpi_set_risk(ndpi_struct, connection, NDPI_TLS_CERTIFICATE_EXPIRED); /* Certificate expired */
	  }
	}
      }
    } else if((packet->payload[i] == 0x55) && (packet->payload[i+1] == 0x1d) && (packet->payload[i+2] == 0x11)) {
      /* Organization OID: 2.5.29.17 (subjectAltName) */
      u_int8_t matched_name = 0;

      /* If the client hello was not observed or the requested name was missing, there is no need to trigger an alert */
      if(connection->host_server_name[0] == '\0')
	matched_name = 1;
	
#ifdef DEBUG_TLS
      printf("******* [TLS] Found subjectAltName\n");
#endif

      i += 3 /* skip the initial patten 55 1D 11 */;

      /* skip the first type, 0x04 == BIT STRING, and jump to it's length */
      if(packet->payload[i] == 0x04) i++; else i += 4; /* 4 bytes, with the last byte set to 04 */
      
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
		  // Parse the payload to extract the DNS name or IP address from the subjectAltName extension.
		  // Convert the extracted address to a string format for IPv4 and IPv6.
		  // Clean up the extracted DNS name and ensure it is printable.
		  // Check if the extracted DNS name matches the host server name.
		  // If a match is found, set the matched_name flag and update the server names list in the flow structure.
		  // If the server names list is not initialized, create a new entry for the current DNS name.
		  // If the server names list already exists, append the current DNS name to it with a comma separator.
		  // Attempt to detect a subprotocol based on the DNS name and update the flow structure accordingly.
		  // <MASK>

		  i += len;
		} else {
#if DEBUG_TLS
		  printf("[TLS] Leftover %u bytes", packet->payload_packet_len - i);
#endif
		  ndpi_set_risk(ndpi_struct, connection, NDPI_TLS_SUSPICIOUS_EXTENSION);
		  break;
		}
	      } else {
		break;
	      }
	    } /* while */

	    if(!matched_name)
	      ndpi_set_risk(ndpi_struct, connection, NDPI_TLS_CERTIFICATE_MISMATCH); /* Certificate mismatch */
	  }
	}
      }
    }
  } /* for */

  if(rdn_len && (connection->protos.tls_quic.subjectDN == NULL)) {
    connection->protos.tls_quic.subjectDN = ndpi_strdup(rdnSeqBuf);

    if(connection->detected_protocol_stack[1] == NDPI_PROTOCOL_UNKNOWN) {
      /* No idea what is happening behind the scenes: let's check the certificate */
      u_int32_t val;
      int rc = ndpi_match_string_value(ndpi_struct->tls_cert_subject_automa.ac_automa,
				       rdnSeqBuf, strlen(rdnSeqBuf), &val);

      if(rc == 0) {
	/* Match found */
	u_int16_t proto_id = (u_int16_t)val;
	ndpi_protocol ret = { NDPI_PROTOCOL_TLS, proto_id, NDPI_PROTOCOL_CATEGORY_UNSPECIFIED};

	connection->detected_protocol_stack[0] = proto_id,
	  connection->detected_protocol_stack[1] = NDPI_PROTOCOL_TLS;

	connection->category = ndpi_get_proto_category(ndpi_struct, ret);
	ndpi_check_subprotocol_risk(ndpi_struct, connection, proto_id);

	if(ndpi_struct->tls_cert_cache == NULL)
	  ndpi_struct->tls_cert_cache = ndpi_lru_cache_init(1024);

	if(ndpi_struct->tls_cert_cache && packet->iph) {
	  u_int32_t key = packet->iph->saddr + packet->tcp->source; /* Server */

	  ndpi_lru_add_to_cache(ndpi_struct->tls_cert_cache, key, proto_id);
	}
      }
    }
  }

  if(connection->protos.tls_quic.subjectDN && connection->protos.tls_quic.issuerDN
     && (!strcmp(connection->protos.tls_quic.subjectDN, connection->protos.tls_quic.issuerDN)))
    ndpi_set_risk(ndpi_struct, connection, NDPI_TLS_SELFSIGNED_CERTIFICATE);

#if DEBUG_TLS
  printf("[TLS] %s() SubjectDN [%s]\n", __FUNCTION__, rdnSeqBuf);
#endif
}
