GF_Err mpgviddmx_process(GF_Filter *filter)
{
	GF_MPGVidDmxCtx *ctx = gf_filter_get_udta(filter);
	GF_FilterPacket *pck, *dst_pck;
	u64 byte_offset;
	s64 vosh_start = -1;
	s64 vosh_end = -1;
	GF_Err e;
	char *data;
	u8 *start;
	u32 pck_size;
	s32 remain;

	//always reparse duration
	if (!ctx->duration.num)
		mpgviddmx_check_dur(filter, ctx);

	pck = gf_filter_pid_get_packet(ctx->ipid);
	if (!pck) {
		if (gf_filter_pid_is_eos(ctx->ipid)) {
			mpgviddmx_enqueue_or_dispatch(ctx, NULL, GF_TRUE, GF_TRUE);
			if (ctx->opid)
				gf_filter_pid_set_eos(ctx->opid);
			if (ctx->src_pck) gf_filter_pck_unref(ctx->src_pck);
			ctx->src_pck = NULL;
			return GF_EOS;
		}
		return GF_OK;
	}

	data = (char *) gf_filter_pck_get_data(pck, &pck_size);
	byte_offset = gf_filter_pck_get_byte_offset(pck);

	start = data;
	remain = pck_size;

	//input pid was muxed - we flushed pending data , update cts
	if (!ctx->resume_from && ctx->timescale) {
		if (!ctx->notime) {
			u64 ts = gf_filter_pck_get_cts(pck);
			if (ts != GF_FILTER_NO_TS) {
				if (!ctx->cts || !ctx->recompute_cts)
					ctx->cts = ts;
			}
			ts = gf_filter_pck_get_dts(pck);
			if (ts != GF_FILTER_NO_TS) {
				if (!ctx->dts || !ctx->recompute_cts) {
					ctx->dts = ts;
				}

				if (!ctx->prev_dts) ctx->prev_dts = ts;
				else if (ctx->prev_dts != ts) {
					u64 diff = ts;
					diff -= ctx->prev_dts;
					if (!ctx->cur_fps.den)
						ctx->cur_fps.den = (u32) diff;
					else if (ctx->cur_fps.den > diff)
						ctx->cur_fps.den = (u32) diff;

					ctx->prev_dts = ts;
				}
			}
		}

		gf_filter_pck_get_framing(pck, &ctx->input_is_au_start, &ctx->input_is_au_end);
		//we force it to true to deal with broken avi packaging where a video AU is split across several AVI video frames
		ctx->input_is_au_end = GF_FALSE;
		//this will force CTS recomput of each frame
		if (ctx->recompute_cts) ctx->input_is_au_start = GF_FALSE;
		if (ctx->src_pck) gf_filter_pck_unref(ctx->src_pck);
		ctx->src_pck = pck;
		gf_filter_pck_ref_props(&ctx->src_pck);
	}

	//we stored some data to find the complete vosh, aggregate this packet with current one
	if (!ctx->resume_from && ctx->hdr_store_size) {
		if (ctx->hdr_store_alloc < ctx->hdr_store_size + pck_size) {
			ctx->hdr_store_alloc = ctx->hdr_store_size + pck_size;
			ctx->hdr_store = gf_realloc(ctx->hdr_store, sizeof(char)*ctx->hdr_store_alloc);
		}
		memcpy(ctx->hdr_store + ctx->hdr_store_size, data, sizeof(char)*pck_size);
		if (byte_offset != GF_FILTER_NO_BO) {
			if (byte_offset >= ctx->hdr_store_size)
				byte_offset -= ctx->hdr_store_size;
			else
				byte_offset = GF_FILTER_NO_BO;
		}
		ctx->hdr_store_size += pck_size;
		start = data = ctx->hdr_store;
		remain = pck_size = ctx->hdr_store_size;
	}

	if (ctx->resume_from) {
		if (gf_filter_pid_would_block(ctx->opid))
			return GF_OK;

		//resume from data copied internally
		if (ctx->hdr_store_size) {
			gf_assert(ctx->resume_from <= ctx->hdr_store_size);
			start = data = ctx->hdr_store + ctx->resume_from;
			remain = pck_size = ctx->hdr_store_size - ctx->resume_from;
		} else {
			gf_assert(remain >= (s32) ctx->resume_from);
			start += ctx->resume_from;
			remain -= ctx->resume_from;
		}
		ctx->resume_from = 0;
	}

	if (!ctx->bs) {
		ctx->bs = gf_bs_new(start, remain, GF_BITSTREAM_READ);
	} else {
		gf_bs_reassign_buffer(ctx->bs, start, remain);
	}
	if (!ctx->vparser) {
		ctx->vparser = gf_m4v_parser_bs_new(ctx->bs, ctx->is_mpg12);
	}


	while (remain) {
		Bool full_frame;
		u8 *packet_data;
		s32 current;
		u8 sc_type, forced_sc_type=0;
		Bool sc_type_forced = GF_FALSE;
		Bool skip_pck = GF_FALSE;
		u8 ftype;
		u32 tinc;
		u64 size=0;
		u64 fstart=0;
		Bool is_coded;
		u32 bytes_from_store = 0;
		u32 hdr_offset = 0;
		Bool copy_last_bytes = GF_FALSE;

		//not enough bytes to parse start code
		if (remain<5) {
			memcpy(ctx->hdr_store, start, remain);
			ctx->bytes_in_header = remain;
			break;
		}
		current = -1;

		//we have some potential bytes of a start code in the store, copy some more bytes and check if valid start code.
		//if not, dispatch these bytes as continuation of the data
		if (ctx->bytes_in_header) {
			//the two zones may overlap
			memmove(ctx->hdr_store + ctx->bytes_in_header, start, MIN_HDR_STORE - ctx->bytes_in_header);
			current = mpgviddmx_next_start_code(ctx->hdr_store, MIN_HDR_STORE);

			//no start code in stored buffer
			if ((current<0) || (current >= (s32) ctx->bytes_in_header) )  {
				if (ctx->opid) {
					dst_pck = gf_filter_pck_new_alloc(ctx->opid, ctx->bytes_in_header, &packet_data);
					if (!dst_pck) return GF_OUT_OF_MEM;

					if (ctx->src_pck) gf_filter_pck_merge_properties(ctx->src_pck, dst_pck);
					gf_filter_pck_set_cts(dst_pck, GF_FILTER_NO_TS);
					gf_filter_pck_set_dts(dst_pck, GF_FILTER_NO_TS);
					gf_filter_pck_set_sap(dst_pck, GF_FILTER_SAP_NONE);
					memcpy(packet_data, ctx->hdr_store, ctx->bytes_in_header);
					gf_filter_pck_set_framing(dst_pck, GF_FALSE, GF_FALSE);

					if (byte_offset != GF_FILTER_NO_BO) {
						gf_filter_pck_set_byte_offset(dst_pck, byte_offset - ctx->bytes_in_header);
					}

					mpgviddmx_enqueue_or_dispatch(ctx, dst_pck, GF_FALSE, GF_FALSE);
				}

				if (current<0) current = -1;
				else current -= ctx->bytes_in_header;
				ctx->bytes_in_header = 0;
			} else {
				//we have a valid start code, check which byte in our store or in the packet payload is the start code type
				//and remember its location to reinit the parser from there
				hdr_offset = 4 - ctx->bytes_in_header + current;
				//bytes still to dispatch
				bytes_from_store = ctx->bytes_in_header;
				ctx->bytes_in_header = 0;
				if (!hdr_offset) {
					forced_sc_type = ctx->hdr_store[current+3];
				} else {
					forced_sc_type = start[hdr_offset-1];
				}
				sc_type_forced = GF_TRUE;
			}
		}
		//no startcode in store, look for startcode in packet
		if (current == -1) {
			//locate next start code
			current = mpgviddmx_next_start_code(start, remain);
			//no start code, dispatch the block
			if (current<0) {
				u8 b3, b2, b1;
				if (! ctx->frame_started) {
					GF_LOG(GF_LOG_DEBUG, GF_LOG_MEDIA, ("[MPGVid] no start code in block and no frame started, discarding data\n" ));
					break;
				}
				size = remain;
				b3 = start[remain-3];
				b2 = start[remain-2];
				b1 = start[remain-1];
				//we may have a startcode at the end of the packet, store it and don't dispatch the last 3 bytes !
				if (!b1 || !b2 || !b3) {
					copy_last_bytes = GF_TRUE;
					gf_assert(size >= 3);
					size -= 3;
					ctx->bytes_in_header = 3;
				}

				dst_pck = gf_filter_pck_new_alloc(ctx->opid, (u32) size, &packet_data);
				if (!dst_pck) return GF_OUT_OF_MEM;

				if (ctx->src_pck) gf_filter_pck_merge_properties(ctx->src_pck, dst_pck);
				memcpy(packet_data, start, (size_t) size);
				gf_filter_pck_set_framing(dst_pck, GF_FALSE, GF_FALSE);
				gf_filter_pck_set_cts(dst_pck, GF_FILTER_NO_TS);
				gf_filter_pck_set_dts(dst_pck, GF_FILTER_NO_TS);

				if (byte_offset != GF_FILTER_NO_BO) {
					gf_filter_pck_set_byte_offset(dst_pck, byte_offset);
				}

				mpgviddmx_enqueue_or_dispatch(ctx, dst_pck, GF_FALSE, GF_FALSE);
				if (copy_last_bytes) {
					memcpy(ctx->hdr_store, start+remain-3, 3);
				}
				break;
			}
		}

		gf_assert(current>=0);

		//if we are in the middle of parsing the vosh, skip over bytes remaining from previous obj not parsed
		if ((vosh_start>=0) && current) {
			gf_assert(remain>=current);
			start += current;
			remain -= current;
			current = 0;
		}
		//also skip if no output pid
		if (!ctx->opid && current) {
			gf_assert(remain>=current);
			start += current;
			remain -= current;
			current = 0;
		}
		//dispatch remaining bytes
		if (current>0) {
			//flush remaining
			dst_pck = gf_filter_pck_new_alloc(ctx->opid, current, &packet_data);
			if (!dst_pck) return GF_OUT_OF_MEM;

			if (ctx->src_pck) gf_filter_pck_merge_properties(ctx->src_pck, dst_pck);
			gf_filter_pck_set_cts(dst_pck, GF_FILTER_NO_TS);
			gf_filter_pck_set_dts(dst_pck, GF_FILTER_NO_TS);
			gf_filter_pck_set_framing(dst_pck, GF_FALSE, GF_FALSE);
			//bytes were partly in store, partly in packet
			if (bytes_from_store) {
				if (byte_offset != GF_FILTER_NO_BO) {
					gf_filter_pck_set_byte_offset(dst_pck, byte_offset - bytes_from_store);
				}
				gf_assert(bytes_from_store>=(u32) current);
				bytes_from_store -= current;
				memcpy(packet_data, ctx->hdr_store, current);
			} else {
				//bytes were only in packet
				if (byte_offset != GF_FILTER_NO_BO) {
					gf_filter_pck_set_byte_offset(dst_pck, byte_offset);
				}
				memcpy(packet_data, start, current);
				gf_assert(remain>=current);
				start += current;
				remain -= current;
				current = 0;
			}
			gf_filter_pck_set_carousel_version(dst_pck, 1);
			mpgviddmx_enqueue_or_dispatch(ctx, dst_pck, GF_FALSE, GF_FALSE);
		}

		//we're align to startcode, stop trashing packets
		ctx->trash_trailer = GF_FALSE;

		//not enough bytes to parse start code
		if (remain<5) {
			memcpy(ctx->hdr_store, start, remain);
			ctx->bytes_in_header = remain;
			break;
		}

		//parse headers
		//we have a start code loaded, eg the data packet does not have a full start code at the beginning
		if (sc_type_forced) {
			gf_bs_reassign_buffer(ctx->bs, start + hdr_offset, remain - hdr_offset);
			sc_type = forced_sc_type;
		} else {
			gf_bs_reassign_buffer(ctx->bs, start, remain);
			gf_bs_read_int(ctx->bs, 24);
			sc_type = gf_bs_read_int(ctx->bs, 8);
		}

		if (ctx->is_mpg12) {
			switch (sc_type) {
			case M2V_SEQ_START_CODE:
			case M2V_EXT_START_CODE:
				gf_bs_reassign_buffer(ctx->bs, start, remain);
				e = gf_m4v_parse_config(ctx->vparser, &ctx->dsi);
				//not enough data, accumulate until we can parse the full header
				if (e==GF_EOS) {
					if (vosh_start<0) vosh_start = 0;
					if (data == ctx->hdr_store) {
						memmove(ctx->hdr_store, start, remain);
						ctx->hdr_store_size = remain;
					} else {
						if (ctx->hdr_store_alloc < ctx->hdr_store_size + pck_size - vosh_start) {
							ctx->hdr_store_alloc = (u32) (ctx->hdr_store_size + pck_size - vosh_start);
							ctx->hdr_store = gf_realloc(ctx->hdr_store, sizeof(char)*ctx->hdr_store_alloc);
						}
						memmove(ctx->hdr_store + ctx->hdr_store_size, data + vosh_start, (size_t) (pck_size - vosh_start) );
						ctx->hdr_store_size += pck_size - (u32) vosh_start;
					}
					gf_filter_pid_drop_packet(ctx->ipid);
					return GF_OK;
				} else if (e != GF_OK) {
					GF_LOG(GF_LOG_ERROR, GF_LOG_MEDIA, ("[MPGVid] Failed to parse VOS header: %s\n", gf_error_to_string(e) ));
				} else if (ctx->dsi.width && ctx->dsi.height) {
					u32 obj_size = (u32) gf_m4v_get_object_start(ctx->vparser);
					if (vosh_start<0) vosh_start = 0;
					vosh_end = start - (u8 *)data + obj_size;
					vosh_end -= vosh_start;
					mpgviddmx_check_pid(filter, ctx,(u32)  vosh_end, data+vosh_start);
				}
				break;
			case M2V_PIC_START_CODE:
				break;
			default:
				break;
			}

		} else {
			u8 PL;
			switch (sc_type) {
			case M4V_VOS_START_CODE:
				ctx->dsi.VideoPL = (u8) gf_bs_read_u8(ctx->bs);
				vosh_start = start - (u8 *)data;
				skip_pck = GF_TRUE;
				gf_assert(remain>=5);
				start += 5;
				remain -= 5;
				break;
			case M4V_VOL_START_CODE:
				gf_bs_reassign_buffer(ctx->bs, start, remain);
				PL = ctx->dsi.VideoPL;
				e = gf_m4v_parse_config(ctx->vparser, &ctx->dsi);
				ctx->dsi.VideoPL = PL;
				//not enough data, accumulate until we can parse the full header
				if (e==GF_EOS) {
					if (vosh_start<0) vosh_start = 0;
					if (data == ctx->hdr_store) {
						memmove(ctx->hdr_store, start, remain);
						ctx->hdr_store_size = remain;
					} else {
						if (ctx->hdr_store_alloc < ctx->hdr_store_size + pck_size - vosh_start) {
							ctx->hdr_store_alloc = (u32) (ctx->hdr_store_size + pck_size - (u32) vosh_start);
							ctx->hdr_store = gf_realloc(ctx->hdr_store, sizeof(char)*ctx->hdr_store_alloc);
						}
						memmove(ctx->hdr_store + ctx->hdr_store_size, data + vosh_start, (size_t) (pck_size - vosh_start) );
						ctx->hdr_store_size += pck_size - (u32) vosh_start;
					}
					gf_filter_pid_drop_packet(ctx->ipid);
					return GF_OK;
				} else if (e != GF_OK) {
					GF_LOG(GF_LOG_ERROR, GF_LOG_MEDIA, ("[MPGVid] Failed to parse VOS header: %s\n", gf_error_to_string(e) ));
				} else {
					u32 obj_size = (u32) gf_m4v_get_object_start(ctx->vparser);
					if (vosh_start<0) vosh_start = 0;
					vosh_end = start - (u8 *)data + obj_size;
					vosh_end -= vosh_start;
					mpgviddmx_check_pid(filter, ctx,(u32)  vosh_end, data+vosh_start);
					skip_pck = GF_TRUE;
					gf_assert(remain>=(s32) obj_size);
					start += obj_size;
					remain -= obj_size;
				}
				break;
			case M4V_VOP_START_CODE:
			case M4V_GOV_START_CODE:
				break;

			case M4V_VO_START_CODE:
			case M4V_VISOBJ_START_CODE:
			default:
				if (vosh_start>=0) {
					skip_pck = GF_TRUE;
					gf_assert(remain>=4);
					start += 4;
					remain -= 4;
				} else if (!ctx->width) {
					gf_bs_reassign_buffer(ctx->bs, start, remain);
					PL = ctx->dsi.VideoPL;
					e = gf_m4v_parse_config(ctx->vparser, &ctx->dsi);
					if (ctx->dsi.width) {
						u32 obj_size = (u32) gf_m4v_get_object_start(ctx->vparser);
						if (vosh_start<0) vosh_start = 0;
						vosh_end = start - (u8 *)data + obj_size;
						vosh_end -= vosh_start;
						mpgviddmx_check_pid(filter, ctx,(u32)  vosh_end, data+vosh_start);
						skip_pck = GF_TRUE;
						gf_assert(remain>=(s32) obj_size);
						start += obj_size;
						remain -= obj_size;
					}
				}
				break;
			}
		}

		if (skip_pck) {
			continue;
		}

		if (!ctx->opid) {
			gf_assert(remain>=4);
			start += 4;
			remain -= 4;
			continue;
		}

		if (!ctx->is_playing) {
			ctx->resume_from = (u32) ((char *)start -  (char *)data);
			return GF_OK;
		}
		//at this point, we no longer reaggregate packets
		ctx->hdr_store_size = 0;

		if (ctx->in_seek) {
			u64 nb_frames_at_seek = (u64) (ctx->start_range * ctx->cur_fps.num);
			if (ctx->cts + ctx->cur_fps.den >= nb_frames_at_seek) {
				//u32 samples_to_discard = (ctx->cts + ctx->dts_inc) - nb_samples_at_seek;
				ctx->in_seek = GF_FALSE;
			}
		}
		//may happen that after all our checks, only 4 bytes are left, continue to store these 4 bytes
		if (remain<5)
			continue;

		//good to go
		gf_m4v_parser_reset(ctx->vparser, sc_type_forced ? forced_sc_type + 1 : 0);
		size = 0;
		e = gf_m4v_parse_frame(ctx->vparser, &ctx->dsi, &ftype, &tinc, &size, &fstart, &is_coded);

		//only for m1v/m2v, for m4v we may have fstart>0 when we strip VO and VISOBJ
		if (ctx->is_mpg12 && fstart && (fstart<remain)) {
			//start code (4 bytes) in header, adjst frame start and size
			if (sc_type_forced) {
				fstart += 4;
				size-=4;
			}
			dst_pck = gf_filter_pck_new_alloc(ctx->opid, (u32) fstart, &packet_data);
			if (!dst_pck) return GF_OUT_OF_MEM;

			if (ctx->src_pck) gf_filter_pck_merge_properties(ctx->src_pck, dst_pck);
			memcpy(packet_data, start, (size_t) fstart);
			gf_filter_pck_set_framing(dst_pck, GF_FALSE, GF_FALSE);
			gf_filter_pck_set_cts(dst_pck, GF_FILTER_NO_TS);
			gf_filter_pck_set_dts(dst_pck, GF_FILTER_NO_TS);

			if (byte_offset != GF_FILTER_NO_BO) {
				gf_filter_pck_set_byte_offset(dst_pck, byte_offset);
				byte_offset+=fstart;
			}

			mpgviddmx_enqueue_or_dispatch(ctx, dst_pck, GF_FALSE, GF_FALSE);

			start += fstart;
			remain -= (s32) fstart;
		}
		//we may have VO or other packets before (fstart is on first of vop/gov/vol/vos)
		else if (fstart && (fstart + size <= remain)) {
			//start code (4 bytes) in header, adjst frame start and size
			if (sc_type_forced) {
				fstart += 4;
				size-=4;
			}
			size += fstart;
		}

		//we skipped bytes already in store + end of start code present in packet, so the size of the first object
		//needs adjustement
		if (bytes_from_store) {
			size += bytes_from_store + hdr_offset;
			if (size > remain) {
				e = GF_NON_COMPLIANT_BITSTREAM;
				GF_LOG(GF_LOG_ERROR, GF_LOG_MEDIA, ("[MPGVid] packet too large to process (size %llu remain %d)\n", size, remain ));
				break;
			}
		}

		if ((e == GF_EOS) && !ctx->input_is_au_end) {
			u8 b3 = start[remain-3];
			u8 b2 = start[remain-2];
			u8 b1 = start[remain-1];

			//we may have a startcode at the end of the packet, store it and don't dispatch the last 3 bytes !
			if (!b1 || !b2 || !b3) {
				copy_last_bytes = GF_TRUE;
				gf_assert(size >= 3);
				size -= 3;
				ctx->bytes_in_header = 3;
			}
			full_frame = GF_FALSE;
		} else {
			full_frame = GF_TRUE;
		}

		if (ftype) {
			if (!is_coded) {
				/*if prev is B and we're parsing a packed bitstream discard n-vop*/
				if ((ctx->forced_packed && ctx->b_frames)
					/*policy is to import at variable frame rate, skip*/
					|| ctx->vfr
				) {
					if (ctx->vfr) {
						ctx->is_vfr = GF_TRUE;
						mpgviddmx_update_time(ctx);
					} else {
						ctx->is_packed = GF_TRUE;
					}

					//part of the frame was in store, adjust size
					if (bytes_from_store)
						size-= bytes_from_store + hdr_offset;

					gf_assert(remain>=size);
					start += size;
					remain -= (s32) size;
					//trash all packets until we align to a new startcode
					ctx->trash_trailer = full_frame ? GF_FALSE : GF_TRUE;
					continue;
				}

				/*policy is to keep non coded frame (constant frame rate), add*/
			}

			if (ftype==3) {
				//count number of B-frames since last ref
				ctx->b_frames++;
				ctx->nb_b++;
			} else {
				//flush all pending packets
				mpgviddmx_enqueue_or_dispatch(ctx, NULL, GF_TRUE, GF_FALSE);
				//remember the CTS of the last ref
				ctx->last_ref_cts = ctx->cts;
				if (ctx->max_b < ctx->b_frames) ctx->max_b = ctx->b_frames;

				ctx->b_frames = 0;
				if (ftype==2)
					ctx->nb_p++;
				else
					ctx->nb_i++;
			}
			ctx->nb_frames++;
		} else {
			full_frame = GF_FALSE;
		}

		dst_pck = gf_filter_pck_new_alloc(ctx->opid, (u32) size, &packet_data);
		if (!dst_pck) return GF_OUT_OF_MEM;

		if (ctx->src_pck) gf_filter_pck_merge_properties(ctx->src_pck, dst_pck);
		//bytes come from both our store and the data packet
		if (bytes_from_store) {
			memcpy(packet_data, ctx->hdr_store+current, bytes_from_store);
			gf_assert(size >= bytes_from_store);
			size -= bytes_from_store;
			if (byte_offset != GF_FILTER_NO_BO) {
				gf_filter_pck_set_byte_offset(dst_pck, byte_offset - bytes_from_store);
			}
			memcpy(packet_data + bytes_from_store, start, (size_t) size);
		} else {
			//bytes only come the data packet
			memcpy(packet_data, start, (size_t) size);
			if (byte_offset != GF_FILTER_NO_BO) {
				gf_filter_pck_set_byte_offset(dst_pck, byte_offset + start - (u8 *) data);
			}
		}

		if (ftype) {
			gf_assert(packet_data[0] == 0);
			gf_assert(packet_data[1] == 0);
			gf_assert(packet_data[2] == 1);

			gf_filter_pck_set_framing(dst_pck, GF_TRUE, (full_frame || ctx->input_is_au_end) ? GF_TRUE : GF_FALSE);
			gf_filter_pck_set_cts(dst_pck, ctx->cts);
			gf_filter_pck_set_dts(dst_pck, ctx->dts);

			if (ctx->input_is_au_start) {
				ctx->input_is_au_start = GF_FALSE;
			} else {
				//we use the carousel flag temporarily to indicate the cts must be recomputed
				gf_filter_pck_set_carousel_version(dst_pck, 1);
			}
			gf_filter_pck_set_sap(dst_pck, (ftype==1) ? GF_FILTER_SAP_1 : GF_FILTER_SAP_NONE);
			if (ctx->cur_fps.den > 0) gf_filter_pck_set_duration(dst_pck, ctx->cur_fps.den);
			if (ctx->in_seek) gf_filter_pck_set_seek_flag(dst_pck, GF_TRUE);
			ctx->frame_started = GF_TRUE;

			mpgviddmx_enqueue_or_dispatch(ctx, dst_pck, GF_FALSE, GF_FALSE);

			mpgviddmx_update_time(ctx);
		} else {
			gf_filter_pck_set_framing(dst_pck, GF_FALSE, (full_frame || ctx->input_is_au_end) ? GF_TRUE : GF_FALSE);
			mpgviddmx_enqueue_or_dispatch(ctx, dst_pck, GF_FALSE, GF_FALSE);
		}

		if (!full_frame) {
			if (copy_last_bytes) {
				memcpy(ctx->hdr_store, start+remain-3, 3);
			}
			break;
		}
		gf_assert(remain>=size);
		start += size;
		remain -= (s32) size;
	}
	gf_filter_pid_drop_packet(ctx->ipid);

	return GF_OK;
}
