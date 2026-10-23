static int _gd2GetHeader(gdIOCtxPtr in, int *sx, int *sy, int *cs, int *vers, int *fmt, int *ncx, int *ncy, t_chunk_info ** chunkIdx)
{
	int i;
	int ch;
	char id[5];
	t_chunk_info *cidx;
	int sidx;
	int nc;

	GD2_DBG(php_gd_error("Reading gd2 header info"));

	for (i = 0; i < 4; i++) {
		ch = gdGetC(in);
		if (ch == EOF) {
			goto fail1;
		}
		id[i] = ch;
	}
	id[4] = 0;

	GD2_DBG(php_gd_error("Got file code: %s", id));

	/* Equiv. of 'magick'.  */
	if (strcmp(id, GD2_ID) != 0) {
		GD2_DBG(php_gd_error("Not a valid gd2 file"));
		goto fail1;
	}

	/* Version */
	if (gdGetWord(vers, in) != 1) {
		goto fail1;
	}
	GD2_DBG(php_gd_error("Version: %d", *vers));

	if ((*vers != 1) && (*vers != 2)) {
		GD2_DBG(php_gd_error("Bad version: %d", *vers));
		goto fail1;
	}

	/* Image Size */
	if (!gdGetWord(sx, in)) {
		GD2_DBG(php_gd_error("Could not get x-size"));
		goto fail1;
	}
	if (!gdGetWord(sy, in)) {
		GD2_DBG(php_gd_error("Could not get y-size"));
		goto fail1;
	}
	GD2_DBG(php_gd_error("Image is %dx%d", *sx, *sy));

	/* Chunk Size (pixels, not bytes!) */
	if (gdGetWord(cs, in) != 1) {
		goto fail1;
	}
	GD2_DBG(php_gd_error("ChunkSize: %d", *cs));

	if ((*cs < GD2_CHUNKSIZE_MIN) || (*cs > GD2_CHUNKSIZE_MAX)) {
		GD2_DBG(php_gd_error("Bad chunk size: %d", *cs));
		goto fail1;
	}

	/* Data Format */
	if (gdGetWord(fmt, in) != 1) {
		goto fail1;
	}
	GD2_DBG(php_gd_error("Format: %d", *fmt));

	if ((*fmt != GD2_FMT_RAW) && (*fmt != GD2_FMT_COMPRESSED) && (*fmt != GD2_FMT_TRUECOLOR_RAW) && (*fmt != GD2_FMT_TRUECOLOR_COMPRESSED)) {
		GD2_DBG(php_gd_error("Bad data format: %d", *fmt));
		goto fail1;
	}

	/* # of chunks wide */
	if (gdGetWord(ncx, in) != 1) {
		goto fail1;
	}
	GD2_DBG(php_gd_error("%d Chunks Wide", *ncx));

	/* # of chunks high */
	if (gdGetWord(ncy, in) != 1) {
		goto fail1;
	}
	GD2_DBG(php_gd_error("%d Chunks vertically", *ncy));

	if (gd2_compressed(*fmt)) {
		nc = (*ncx) * (*ncy);
		GD2_DBG(php_gd_error("Reading %d chunk index entries", nc));
		sidx = sizeof(t_chunk_info) * nc;
		if (sidx <= 0) {
			goto fail1;
		}
		cidx = gdCalloc(sidx, 1);
		for (i = 0; i < nc; i++) {
			if (gdGetInt(&cidx[i].offset, in) != 1) {
				gdFree(cidx);
				goto fail1;
			}
			if (gdGetInt(&cidx[i].size, in) != 1) {
				gdFree(cidx);
				goto fail1;
			}
			if (cidx[i].offset < 0 || cidx[i].size < 0) {
				gdFree(cidx);
				goto fail1;
			}
		}
		*chunkIdx = cidx;
	}

	GD2_DBG(php_gd_error("gd2 header complete"));

	return 1;

fail1:
	return 0;
}
