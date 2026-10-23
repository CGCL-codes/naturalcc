BGD_DECLARE(gdImagePtr) gdImageCreateFromGd2Ctx (gdIOCtxPtr in)
{
	int sx, sy;
	int i;
	int ncx, ncy, nc, cs, cx, cy;
	int x, y, ylo, yhi, xlo, xhi;
	int vers, fmt;
	t_chunk_info *chunkIdx = NULL;	/* So we can gdFree it with impunity. */
	unsigned char *chunkBuf = NULL;	/* So we can gdFree it with impunity. */
	int chunkNum = 0;
	int chunkMax = 0;
	uLongf chunkLen;
	int chunkPos = 0;
	int compMax = 0;
	int bytesPerPixel;
	char *compBuf = NULL;		/* So we can gdFree it with impunity. */

	gdImagePtr im;

	/* Get the header */
	im =
	    _gd2CreateFromFile (in, &sx, &sy, &cs, &vers, &fmt, &ncx, &ncy,
	                        &chunkIdx);
	if (im == NULL) {
		/* No need to free chunkIdx as _gd2CreateFromFile does it for us. */
		return 0;
	}

	bytesPerPixel = im->trueColor ? 4 : 1;
	nc = ncx * ncy;

	if (gd2_compressed (fmt)) {
		/* Find the maximum compressed chunk size. */
		compMax = 0;
		for (i = 0; (i < nc); i++) {
			if (chunkIdx[i].size > compMax) {
				compMax = chunkIdx[i].size;
			};
		};
		compMax++;

		/* Allocate buffers */
		chunkMax = cs * bytesPerPixel * cs;
		chunkBuf = gdCalloc (chunkMax, 1);
		if (!chunkBuf) {
			goto fail;
		}
		compBuf = gdCalloc (compMax, 1);
		if (!compBuf) {
			goto fail;
		}

		GD2_DBG (printf ("Largest compressed chunk is %d bytes\n", compMax));
	};

	/*      if ( (ncx != sx / cs) || (ncy != sy / cs)) { */
	/*              goto fail2; */
	/*      }; */

	/* Read the data... */
	for (cy = 0; (cy < ncy); cy++) {
		for (cx = 0; (cx < ncx); cx++) {

			ylo = cy * cs;
			yhi = ylo + cs;
			if (yhi > im->sy) {
				yhi = im->sy;
			};

			GD2_DBG (printf
			         ("Processing Chunk %d (%d, %d), y from %d to %d\n",
			          chunkNum, cx, cy, ylo, yhi));

			if (gd2_compressed (fmt)) {

				chunkLen = chunkMax;

				if (!_gd2ReadChunk (chunkIdx[chunkNum].offset,
				                    compBuf,
				                    chunkIdx[chunkNum].size,
				                    (char *) chunkBuf, &chunkLen, in)) {
					GD2_DBG (printf ("Error reading comproessed chunk\n"));
					goto fail;
				};

				chunkPos = 0;
			};

			for (y = ylo; (y < yhi); y++) {

				xlo = cx * cs;
				xhi = xlo + cs;
				if (xhi > im->sx) {
					xhi = im->sx;
				};
				/*GD2_DBG(printf("y=%d: ",y)); */
				if (!gd2_compressed (fmt)) {
					for (x = xlo; x < xhi; x++) {

						if (im->trueColor) {
							if (!gdGetInt (&im->tpixels[y][x], in)) {
								/*printf("EOF while reading\n"); */
								/*gdImageDestroy(im); */
								/*return 0; */
								im->tpixels[y][x] = 0;
							}
						} else {
							int ch;
							if (!gdGetByte (&ch, in)) {
								/*printf("EOF while reading\n"); */
								/*gdImageDestroy(im); */
								/*return 0; */
								ch = 0;
							}
							im->pixels[y][x] = ch;
						}
					}
				} else {
					for (x = xlo; x < xhi; x++) {
						if (im->trueColor) {
							/* 2.0.1: work around a gcc bug by being verbose.
							   TBB */
							int a = chunkBuf[chunkPos++] << 24;
							int r = chunkBuf[chunkPos++] << 16;
							int g = chunkBuf[chunkPos++] << 8;
							int b = chunkBuf[chunkPos++];
							/* 2.0.11: tpixels */
							im->tpixels[y][x] = a + r + g + b;
						} else {
							im->pixels[y][x] = chunkBuf[chunkPos++];
						}
					};
				};
				/*GD2_DBG(printf("\n")); */
			};
			chunkNum++;
		};
	};

	GD2_DBG (printf ("Freeing memory\n"));

	gdFree (chunkBuf);
	gdFree (compBuf);
	gdFree (chunkIdx);

	GD2_DBG (printf ("Done\n"));

	return im;

fail:
	gdImageDestroy (im);
	if (chunkBuf) {
		gdFree (chunkBuf);
	}
	if (compBuf) {
		gdFree (compBuf);
	}
	if (chunkIdx) {
		gdFree (chunkIdx);
	}
	return 0;
}
