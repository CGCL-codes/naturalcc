static Image *ReadOneMNGImage(MngInfo* mng_data,const ImageInfo *image_info,
  ExceptionInfo *exception)
{
  char
    page_geometry[MagickPathExtent];

  Image
    *image;

  MagickBooleanType
    logging;

  volatile int
    first_mng_object,
    object_id,
    term_chunk_found,
    skip_to_iend;

  volatile ssize_t
    image_count=0;

  MagickBooleanType
    status;

  MagickOffsetType
    offset;

  MngBox
    default_fb,
    fb,
    previous_fb;

#if defined(MNG_INSERT_LAYERS)
  PixelInfo
    mng_background_color;
#endif

  ssize_t
    i;

  size_t
    count;

  ssize_t
    loop_level;

  unsigned char
    *p;

  volatile short
    skipping_loop;

#if defined(MNG_INSERT_LAYERS)
  unsigned int
    mandatory_back=0;
#endif

  volatile unsigned int
#ifdef MNG_OBJECT_BUFFERS
    mng_background_object=0,
#endif
    mng_type=0;   /* 0: PNG or JNG; 1: MNG; 2: MNG-LC; 3: MNG-VLC */

  size_t
    default_frame_timeout,
    frame_timeout,
#if defined(MNG_INSERT_LAYERS)
    image_height,
    image_width,
#endif
    length;

  /* These delays are all measured in image ticks_per_second,
   * not in MNG ticks_per_second
   */
  volatile size_t
    default_frame_delay,
    final_delay,
    final_image_delay,
    frame_delay,
#if defined(MNG_INSERT_LAYERS)
    insert_layers,
#endif
    mng_iterations=1,
    simplicity=0,
    subframe_height=0,
    subframe_width=0;

  previous_fb.top=0;
  previous_fb.bottom=0;
  previous_fb.left=0;
  previous_fb.right=0;
  default_fb.top=0;
  default_fb.bottom=0;
  default_fb.left=0;
  default_fb.right=0;

  logging=LogMagickEvent(CoderEvent,GetMagickModule(),
    "  Enter ReadOneMNGImage()");

  image=mng_data->image;

  if (LocaleCompare(image_info->magick,"MNG") == 0)
    {
      char
        magic_number[MagickPathExtent];

      /* Verify MNG signature.  */
      count=(size_t) ReadBlob(image,8,(unsigned char *) magic_number);
      if ((count < 8) || (memcmp(magic_number,"\212MNG\r\n\032\n",8) != 0))
        ThrowReaderException(CorruptImageError,"ImproperImageHeader");

      /* Initialize some nonzero members of the MngInfo structure.  */
      for (i=0; i < MNG_MAX_OBJECTS; i++)
      {
        mng_data->object_clip[i].right=(ssize_t) PNG_UINT_31_MAX;
        mng_data->object_clip[i].bottom=(ssize_t) PNG_UINT_31_MAX;
      }
      mng_data->exists[0]=MagickTrue;
    }

  skipping_loop=(-1);
  first_mng_object=MagickTrue;
  mng_type=0;
#if defined(MNG_INSERT_LAYERS)
  insert_layers=MagickFalse; /* should be False during convert or mogrify */
#endif
  default_frame_delay=0;
  default_frame_timeout=0;
  frame_delay=0;
  final_delay=1;
  mng_data->ticks_per_second=1UL*image->ticks_per_second;
  object_id=0;
  skip_to_iend=MagickFalse;
  term_chunk_found=MagickFalse;
  mng_data->framing_mode=1;
#if defined(MNG_INSERT_LAYERS)
  mandatory_back=MagickFalse;
#endif
#if defined(MNG_INSERT_LAYERS)
  mng_background_color=image->background_color;
#endif
  default_fb=mng_data->frame;
  previous_fb=mng_data->frame;
  do
  {
    char
      type[MagickPathExtent];

    if (LocaleCompare(image_info->magick,"MNG") == 0)
      {
        unsigned char
          *chunk;

        /*
          Read a new chunk.
        */
        type[0]='\0';
        (void) ConcatenateMagickString(type,"errr",MagickPathExtent);
        length=(size_t) ReadBlobMSBLong(image);
        count=(size_t) ReadBlob(image,4,(unsigned char *) type);

        if (logging != MagickFalse)
          (void) LogMagickEvent(CoderEvent,GetMagickModule(),
           "  Reading MNG chunk type %c%c%c%c, length: %.20g",
           type[0],type[1],type[2],type[3],(double) length);

        if ((length > PNG_UINT_31_MAX) || (length > GetBlobSize(image)) ||
            (count < 4))
          ThrowReaderException(CorruptImageError,"CorruptImage");

        p=NULL;
        chunk=(unsigned char *) NULL;

        if (length != 0)
          {
            chunk=(unsigned char *) AcquireQuantumMemory(length,sizeof(*chunk));

            if (chunk == (unsigned char *) NULL)
              ThrowReaderException(ResourceLimitError,"MemoryAllocationFailed");

            for (i=0; i < (ssize_t) length; i++)
            {
              int
                c;

              c=ReadBlobByte(image);
              if (c == EOF)
                {
                  chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                  ThrowReaderException(CorruptImageError,
                    "InsufficientImageDataInFile");
                }
              chunk[i]=(unsigned char) c;
            }

            p=chunk;
          }

        (void) ReadBlobMSBLong(image);  /* read crc word */

#if !defined(JNG_SUPPORTED)
        if (memcmp(type,mng_JHDR,4) == 0)
          {
            skip_to_iend=MagickTrue;

            if (mng_data->jhdr_warning == 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"JNGCompressNotSupported","`%s'",image->filename);

            mng_data->jhdr_warning++;
          }
#endif
        if (memcmp(type,mng_DHDR,4) == 0)
          {
            skip_to_iend=MagickTrue;

            if (mng_data->dhdr_warning == 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"DeltaPNGNotSupported","`%s'",image->filename);

            mng_data->dhdr_warning++;
          }
        if (memcmp(type,mng_MEND,4) == 0)
          {
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            break;
          }

        if (skip_to_iend)
          {
            if (memcmp(type,mng_IEND,4) == 0)
              skip_to_iend=MagickFalse;

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);

            if (logging != MagickFalse)
              (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                "  Skip to IEND.");

            continue;
          }

        if (memcmp(type,mng_MHDR,4) == 0)
          {
            if (length != 28)
              {
                chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                ThrowReaderException(CorruptImageError,"CorruptImage");
              }

            mng_data->mng_width=(unsigned long)mng_get_long(p);
            mng_data->mng_height=(unsigned long)mng_get_long(&p[4]);

            if (logging != MagickFalse)
              {
                (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                  "  MNG width: %.20g",(double) mng_data->mng_width);
                (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                  "  MNG height: %.20g",(double) mng_data->mng_height);
              }

            p+=8;
            mng_data->ticks_per_second=(size_t) mng_get_long(p);

            if (mng_data->ticks_per_second == 0)
              default_frame_delay=0;

            else
              default_frame_delay=1UL*image->ticks_per_second/
                mng_data->ticks_per_second;

            frame_delay=default_frame_delay;
            simplicity=0;

            p+=16;
            simplicity=(size_t) mng_get_long(p);

            mng_type=1;    /* Full MNG */

            if ((simplicity != 0) && ((simplicity | 11) == 11))
              mng_type=2; /* LC */

            if ((simplicity != 0) && ((simplicity | 9) == 9))
              mng_type=3; /* VLC */

#if defined(MNG_INSERT_LAYERS)
            if (mng_type != 3)
              insert_layers=MagickTrue;
#endif
            if (GetAuthenticPixelQueue(image) != (Quantum *) NULL)
              {
                /* Allocate next image structure.  */
                AcquireNextImage(image_info,image,exception);

                if (GetNextImageInList(image) == (Image *) NULL)
                  return((Image *) NULL);

                image=SyncNextImageInList(image);
                mng_data->image=image;
              }

            if ((mng_data->mng_width > 65535L) ||
                (mng_data->mng_height > 65535L))
              {
                chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                ThrowReaderException(ImageError,"WidthOrHeightExceedsLimit");
              }

            (void) FormatLocaleString(page_geometry,MagickPathExtent,
              "%.20gx%.20g+0+0",(double) mng_data->mng_width,(double)
              mng_data->mng_height);

            mng_data->frame.left=0;
            mng_data->frame.right=(ssize_t) mng_data->mng_width;
            mng_data->frame.top=0;
            mng_data->frame.bottom=(ssize_t) mng_data->mng_height;
            mng_data->clip=default_fb=previous_fb=mng_data->frame;

            for (i=0; i < MNG_MAX_OBJECTS; i++)
              mng_data->object_clip[i]=mng_data->frame;

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_TERM,4) == 0)
          {
            int
              repeat=0;

            if (length != 0)
              repeat=p[0];

            if (repeat == 3 && length > 9)
              {
                final_delay=(png_uint_32) mng_get_long(&p[2]);
                mng_iterations=(png_uint_32) mng_get_long(&p[6]);

                if (mng_iterations == PNG_UINT_31_MAX)
                  mng_iterations=0;

                image->iterations=mng_iterations;
                term_chunk_found=MagickTrue;
              }

            if (logging != MagickFalse)
              {
                (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                  "    repeat=%d,  final_delay=%.20g,  iterations=%.20g",
                  repeat,(double) final_delay, (double) image->iterations);
              }

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }
        if (memcmp(type,mng_DEFI,4) == 0)
          {
            if (mng_type == 3)
              {
                (void) ThrowMagickException(exception,GetMagickModule(),
                  CoderError,"DEFI chunk found in MNG-VLC datastream","`%s'",
                  image->filename);
                chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                continue;
              }

            if (length < 2)
              {
                chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                ThrowReaderException(CorruptImageError,"CorruptImage");
              }

            object_id=((unsigned int) p[0] << 8) | (unsigned int) p[1];

            if (mng_type == 2 && object_id != 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"Nonzero object_id in MNG-LC datastream","`%s'",
                image->filename);

            if (object_id >= MNG_MAX_OBJECTS)
              {
                /*
                  Instead of using a warning we should allocate a larger
                  MngInfo structure and continue.
                */
                (void) ThrowMagickException(exception,GetMagickModule(),
                  CoderError,"object id too large","`%s'",image->filename);
                object_id=MNG_MAX_OBJECTS-1;
              }

            if (mng_data->exists[object_id])
              if (mng_data->frozen[object_id])
                {
                  chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                  (void) ThrowMagickException(exception,
                    GetMagickModule(),CoderError,
                    "DEFI cannot redefine a frozen MNG object","`%s'",
                    image->filename);
                  continue;
                }

            mng_data->exists[object_id]=MagickTrue;

            if (length > 2)
              mng_data->invisible[object_id]=p[2];

            /*
              Extract object offset info.
            */
            if (length > 11)
              {
                mng_data->x_off[object_id]=(ssize_t) mng_get_long(&p[4]);
                mng_data->y_off[object_id]=(ssize_t) mng_get_long(&p[8]);
                if (logging != MagickFalse)
                  {
                    (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                      "  x_off[%d]: %.20g,  y_off[%d]: %.20g",
                      object_id,(double) mng_data->x_off[object_id],
                      object_id,(double) mng_data->y_off[object_id]);
                  }
              }

            /*
              Extract object clipping info.
            */
            if (length > 27)
              mng_data->object_clip[object_id]=mng_read_box(mng_data->frame,0,
                &p[12]);

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }
        if (memcmp(type,mng_bKGD,4) == 0)
          {
            mng_data->have_global_bkgd=MagickFalse;

            if (length > 5)
              {
                mng_data->mng_global_bkgd.red=
                  ScaleShortToQuantum((unsigned short) ((p[0] << 8) | p[1]));

                mng_data->mng_global_bkgd.green=
                  ScaleShortToQuantum((unsigned short) ((p[2] << 8) | p[3]));

                mng_data->mng_global_bkgd.blue=
                  ScaleShortToQuantum((unsigned short) ((p[4] << 8) | p[5]));

                mng_data->have_global_bkgd=MagickTrue;
              }

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }
        if (memcmp(type,mng_BACK,4) == 0)
          {
#if defined(MNG_INSERT_LAYERS)
            if (length > 6)
              mandatory_back=p[6];

            else
              mandatory_back=0;

            if (mandatory_back && length > 5)
              {
                mng_background_color.red=
                    ScaleShortToQuantum((unsigned short) ((p[0] << 8) | p[1]));

                mng_background_color.green=
                    ScaleShortToQuantum((unsigned short) ((p[2] << 8) | p[3]));

                mng_background_color.blue=
                    ScaleShortToQuantum((unsigned short) ((p[4] << 8) | p[5]));

                mng_background_color.alpha=OpaqueAlpha;
              }

#ifdef MNG_OBJECT_BUFFERS
            if (length > 8)
              mng_background_object=(p[7] << 8) | p[8];
#endif
#endif
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_PLTE,4) == 0)
          {
            /* Read global PLTE.  */

            if (length && (length < 769))
              {
                /* Read global PLTE.  */

                if (mng_data->global_plte == (png_colorp) NULL)
                  mng_data->global_plte=(png_colorp) AcquireQuantumMemory(256,
                    sizeof(*mng_data->global_plte));

                if (mng_data->global_plte == (png_colorp) NULL)
                  {
                    mng_data->global_plte_length=0;
                    chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                    ThrowReaderException(ResourceLimitError,
                      "MemoryAllocationFailed");
                  }

                for (i=0; i < (ssize_t) (length/3); i++)
                {
                  mng_data->global_plte[i].red=p[3*i];
                  mng_data->global_plte[i].green=p[3*i+1];
                  mng_data->global_plte[i].blue=p[3*i+2];
                }

                mng_data->global_plte_length=(unsigned int) (length/3);
              }
#ifdef MNG_LOOSE
            for ( ; i < 256; i++)
            {
              mng_data->global_plte[i].red=i;
              mng_data->global_plte[i].green=i;
              mng_data->global_plte[i].blue=i;
            }

            if (length != 0)
              mng_data->global_plte_length=256;
#endif
            else
              mng_data->global_plte_length=0;

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_tRNS,4) == 0)
          {
            /* read global tRNS */

            if (length > 0 && length < 257)
              for (i=0; i < (ssize_t) length; i++)
                mng_data->global_trns[i]=p[i];

#ifdef MNG_LOOSE
            for ( ; i < 256; i++)
              mng_data->global_trns[i]=255;
#endif
            mng_data->global_trns_length=(unsigned int) length;
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }
        if (memcmp(type,mng_gAMA,4) == 0)
          {
            if (length == 4)
              {
                ssize_t
                  igamma;

                igamma=mng_get_long(p);
                mng_data->global_gamma=((float) igamma)*0.00001;
                mng_data->have_global_gama=MagickTrue;
              }

            else
              mng_data->have_global_gama=MagickFalse;

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_cHRM,4) == 0)
          {
            /* Read global cHRM */

            if (length == 32)
              {
                mng_data->global_chrm.white_point.x=0.00001*mng_get_long(p);
                mng_data->global_chrm.white_point.y=0.00001*mng_get_long(&p[4]);
                mng_data->global_chrm.red_primary.x=0.00001*mng_get_long(&p[8]);
                mng_data->global_chrm.red_primary.y=0.00001*
                  mng_get_long(&p[12]);
                mng_data->global_chrm.green_primary.x=0.00001*
                  mng_get_long(&p[16]);
                mng_data->global_chrm.green_primary.y=0.00001*
                  mng_get_long(&p[20]);
                mng_data->global_chrm.blue_primary.x=0.00001*
                  mng_get_long(&p[24]);
                mng_data->global_chrm.blue_primary.y=0.00001*
                  mng_get_long(&p[28]);
                mng_data->have_global_chrm=MagickTrue;
              }
            else
              mng_data->have_global_chrm=MagickFalse;

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_sRGB,4) == 0)
          {
            /*
              Read global sRGB.
            */
            if (length != 0)
              {
                mng_data->global_srgb_intent=
                  Magick_RenderingIntent_from_PNG_RenderingIntent(p[0]);
                mng_data->have_global_srgb=MagickTrue;
              }
            else
              mng_data->have_global_srgb=MagickFalse;

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_iCCP,4) == 0)
          {
            /* To do: */

            /*
              Read global iCCP.
            */
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);

            continue;
          }

        if (memcmp(type,mng_FRAM,4) == 0)
          {
            if (mng_type == 3)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"FRAM chunk found in MNG-VLC datastream","`%s'",
                image->filename);

            if ((mng_data->framing_mode == 2) || (mng_data->framing_mode == 4))
              image->delay=frame_delay;

            frame_delay=default_frame_delay;
            frame_timeout=default_frame_timeout;
            fb=default_fb;

            if (length != 0)
              if (p[0])
                mng_data->framing_mode=p[0];

            if (logging != MagickFalse)
              (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                "    Framing_mode=%d",mng_data->framing_mode);

            if (length > 6)
              {
                /* Note the delay and frame clipping boundaries.  */

                p++; /* framing mode */

                while (((p-chunk) < (long) length) && *p)
                  p++;  /* frame name */

                p++;  /* frame name terminator */

                if ((p-chunk) < (ssize_t) (length-4))
                  {
                    int
                      change_delay,
                      change_timeout,
                      change_clipping;

                    change_delay=(*p++);
                    change_timeout=(*p++);
                    change_clipping=(*p++);
                    p++; /* change_sync */

                    if (change_delay && ((p-chunk) < (ssize_t) (length-4)))
                      {
                        frame_delay=1UL*image->ticks_per_second*
                          mng_get_long(p);

                        if (mng_data->ticks_per_second != 0)
                          frame_delay/=mng_data->ticks_per_second;

                        else
                          frame_delay=PNG_UINT_31_MAX;

                        if (change_delay == 2)
                          default_frame_delay=frame_delay;

                        p+=4;

                        if (logging != MagickFalse)
                          (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                            "    Framing_delay=%.20g",(double) frame_delay);
                      }

                    if (change_timeout && ((p-chunk) < (ssize_t) (length-4)))
                      {
                        frame_timeout=1UL*image->ticks_per_second*
                          mng_get_long(p);

                        if (mng_data->ticks_per_second != 0)
                          frame_timeout/=mng_data->ticks_per_second;

                        else
                          frame_timeout=PNG_UINT_31_MAX;

                        if (change_timeout == 2)
                          default_frame_timeout=frame_timeout;

                        p+=4;

                        if (logging != MagickFalse)
                          (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                            "    Framing_timeout=%.20g",(double) frame_timeout);
                      }

                    if (change_clipping && ((p-chunk) < (ssize_t) (length-16)))
                      {
                        fb=mng_read_box(previous_fb,(char) p[0],&p[1]);
                        p+=16;
                        previous_fb=fb;

                        if (logging != MagickFalse)
                          (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                            "    Frame_clip: L=%.20g R=%.20g T=%.20g B=%.20g",
                            (double) fb.left,(double) fb.right,(double) fb.top,
                            (double) fb.bottom);

                        if (change_clipping == 2)
                          default_fb=fb;
                      }
                  }
              }
            mng_data->clip=fb;
            mng_data->clip=mng_minimum_box(fb,mng_data->frame);

            subframe_width=(size_t) (mng_data->clip.right
               -mng_data->clip.left);

            subframe_height=(size_t) (mng_data->clip.bottom
               -mng_data->clip.top);
            /*
              Insert a background layer behind the frame if framing_mode is 4.
            */
#if defined(MNG_INSERT_LAYERS)
            if (logging != MagickFalse)
              (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                "   subframe_width=%.20g, subframe_height=%.20g",(double)
                subframe_width,(double) subframe_height);

            if (insert_layers && (mng_data->framing_mode == 4) &&
                (subframe_width) && (subframe_height))
              {
                /* Allocate next image structure.  */
                if (GetAuthenticPixelQueue(image) != (Quantum *) NULL)
                  {
                    AcquireNextImage(image_info,image,exception);

                    if (GetNextImageInList(image) == (Image *) NULL)
                      return(DestroyImageList(image));

                    image=SyncNextImageInList(image);
                  }

                mng_data->image=image;

                if (term_chunk_found)
                  {
                    image->start_loop=MagickTrue;
                    image->iterations=mng_iterations;
                    term_chunk_found=MagickFalse;
                  }

                else
                    image->start_loop=MagickFalse;

                image->columns=subframe_width;
                image->rows=subframe_height;
                image->page.width=subframe_width;
                image->page.height=subframe_height;
                image->page.x=mng_data->clip.left;
                image->page.y=mng_data->clip.top;
                image->background_color=mng_background_color;
                image->alpha_trait=UndefinedPixelTrait;
                image->delay=0;
                if (SetImageBackgroundColor(image,exception) == MagickFalse)
                  {
                    chunk=(unsigned char *) RelinquishMagickMemory(chunk);
                    return(DestroyImageList(image));
                  }
                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "  Insert backgd layer, L=%.20g, R=%.20g T=%.20g, B=%.20g",
                    (double) mng_data->clip.left,
                    (double) mng_data->clip.right,
                    (double) mng_data->clip.top,
                    (double) mng_data->clip.bottom);
              }
#endif
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_CLIP,4) == 0)
          {
            unsigned int
              first_object,
              last_object;

            /*
              Read CLIP.
            */
            if (length > 3)
              {
                first_object=(p[0] << 8) | p[1];
                last_object=(p[2] << 8) | p[3];
                p+=4;

                for (i=(int) first_object; i <= (int) last_object; i++)
                {
                  if ((i < 0) || (i >= MNG_MAX_OBJECTS))
                    continue;

                  if (mng_data->exists[i] && !mng_data->frozen[i])
                    {
                      MngBox
                        box;

                      box=mng_data->object_clip[i];
                      if ((p-chunk) < (ssize_t) (length-17))
                        mng_data->object_clip[i]=
                           mng_read_box(box,(char) p[0],&p[1]);
                    }
                }

              }
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_SAVE,4) == 0)
          {
            for (i=1; i < MNG_MAX_OBJECTS; i++)
              if (mng_data->exists[i])
                {
                 mng_data->frozen[i]=MagickTrue;
#ifdef MNG_OBJECT_BUFFERS
                 if (mng_data->ob[i] != (MngBuffer *) NULL)
                    mng_data->ob[i]->frozen=MagickTrue;
#endif
                }

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);

            continue;
          }

        if ((memcmp(type,mng_DISC,4) == 0) || (memcmp(type,mng_SEEK,4) == 0))
          {
            /* Read DISC or SEEK.  */

            if ((length == 0) || (length % 2) || !memcmp(type,mng_SEEK,4))
              {
                for (i=1; i < MNG_MAX_OBJECTS; i++)
                  MngInfoDiscardObject(mng_data,i);
              }

            else
              {
                ssize_t
                  j;

                for (j=1; j < (ssize_t) length; j+=2)
                {
                  i=p[j-1] << 8 | p[j];
                  MngInfoDiscardObject(mng_data,i);
                }
              }

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);

            continue;
          }

        if (memcmp(type,mng_MOVE,4) == 0)
          {
            size_t
              first_object,
              last_object;

            /* read MOVE */

            if (length > 3)
            {
              first_object=(p[0] << 8) | p[1];
              last_object=(p[2] << 8) | p[3];
              p+=4;

              for (i=(ssize_t) first_object; i <= (ssize_t) last_object; i++)
              {
                if ((i < 0) || (i >= MNG_MAX_OBJECTS))
                  continue;

                if (mng_data->exists[i] && !mng_data->frozen[i] &&
                    (p-chunk) < (ssize_t) (length-8))
                  {
                    MngPair
                      new_pair;

                    MngPair
                      old_pair;

                    old_pair.a=mng_data->x_off[i];
                    old_pair.b=mng_data->y_off[i];
                    new_pair=mng_read_pair(old_pair,(int) p[0],&p[1]);
                    mng_data->x_off[i]=new_pair.a;
                    mng_data->y_off[i]=new_pair.b;
                  }
              }
            }

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_LOOP,4) == 0)
          {
            ssize_t loop_iters=1;
            if (length > 4)
              {
                loop_level=chunk[0];
                mng_data->loop_active[loop_level]=1;  /* mark loop active */

                /* Record starting point.  */
                loop_iters=mng_get_long(&chunk[1]);

                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "  LOOP level %.20g has %.20g iterations ",
                    (double) loop_level, (double) loop_iters);

                if (loop_iters <= 0)
                  skipping_loop=loop_level;

                else
                  {
                    if ((MagickSizeType) loop_iters > GetMagickResourceLimit(ListLengthResource))
                      loop_iters=GetMagickResourceLimit(ListLengthResource);
                    if (loop_iters >= 2147483647L)
                      loop_iters=2147483647L;
                    if (image_info->number_scenes != 0)
                      if (loop_iters > (ssize_t) image_info->number_scenes)
                        loop_iters=image_info->number_scenes;
                    mng_data->loop_jump[loop_level]=TellBlob(image);
                    mng_data->loop_count[loop_level]=loop_iters;
                  }

                mng_data->loop_iteration[loop_level]=0;
              }
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_ENDL,4) == 0)
          {
            if (length > 0)
              {
                loop_level=chunk[0];

                if (skipping_loop > 0)
                  {
                    if (skipping_loop == loop_level)
                      {
                        /*
                          Found end of zero-iteration loop.
                        */
                        skipping_loop=(-1);
                        mng_data->loop_active[loop_level]=0;
                      }
                  }

                else
                  {
                    if (mng_data->loop_active[loop_level] == 1)
                      {
                        mng_data->loop_count[loop_level]--;
                        mng_data->loop_iteration[loop_level]++;

                        if (logging != MagickFalse)
                          (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                          "  ENDL: LOOP level %.20g has %.20g remaining iters",
                            (double) loop_level,(double)
                            mng_data->loop_count[loop_level]);

                        if (mng_data->loop_count[loop_level] > 0)
                          {
                            offset=
                              SeekBlob(image,mng_data->loop_jump[loop_level],
                              SEEK_SET);

                            if (offset < 0)
                              {
                                chunk=(unsigned char *) RelinquishMagickMemory(
                                  chunk);
                                ThrowReaderException(CorruptImageError,
                                  "ImproperImageHeader");
                              }
                          }

                        else
                          {
                            short
                              last_level;

                            /*
                              Finished loop.
                            */
                            mng_data->loop_active[loop_level]=0;
                            last_level=(-1);
                            for (i=0; i < loop_level; i++)
                              if (mng_data->loop_active[i] == 1)
                                last_level=(short) i;
                            loop_level=last_level;
                          }
                      }
                  }
              }

            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_CLON,4) == 0)
          {
            if (mng_data->clon_warning == 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"CLON is not implemented yet","`%s'",
                image->filename);

            mng_data->clon_warning++;
          }

        if (memcmp(type,mng_MAGN,4) == 0)
          {
            png_uint_16
              magn_first,
              magn_last,
              magn_mb,
              magn_ml,
              magn_mr,
              magn_mt,
              magn_mx,
              magn_my,
              magn_methx,
              magn_methy;

            if (length > 1)
              magn_first=(p[0] << 8) | p[1];

            else
              magn_first=0;

            if (length > 3)
              magn_last=(p[2] << 8) | p[3];

            else
              magn_last=magn_first;
#ifndef MNG_OBJECT_BUFFERS
            if (magn_first || magn_last)
              if (mng_data->magn_warning == 0)
                {
                  (void) ThrowMagickException(exception,
                     GetMagickModule(),CoderError,
                     "MAGN is not implemented yet for nonzero objects",
                     "`%s'",image->filename);

                   mng_data->magn_warning++;
                }
#endif
            if (length > 4)
              magn_methx=p[4];

            else
              magn_methx=0;

            if (length > 6)
              magn_mx=(p[5] << 8) | p[6];

            else
              magn_mx=1;

            if (magn_mx == 0)
              magn_mx=1;

            if (length > 8)
              magn_my=(p[7] << 8) | p[8];

            else
              magn_my=magn_mx;

            if (magn_my == 0)
              magn_my=1;

            if (length > 10)
              magn_ml=(p[9] << 8) | p[10];

            else
              magn_ml=magn_mx;

            if (magn_ml == 0)
              magn_ml=1;

            if (length > 12)
              magn_mr=(p[11] << 8) | p[12];

            else
              magn_mr=magn_mx;

            if (magn_mr == 0)
              magn_mr=1;

            if (length > 14)
              magn_mt=(p[13] << 8) | p[14];

            else
              magn_mt=magn_my;

            if (magn_mt == 0)
              magn_mt=1;

            if (length > 16)
              magn_mb=(p[15] << 8) | p[16];

            else
              magn_mb=magn_my;

            if (magn_mb == 0)
              magn_mb=1;

            if (length > 17)
              magn_methy=p[17];

            else
              magn_methy=magn_methx;


            if (magn_methx > 5 || magn_methy > 5)
              if (mng_data->magn_warning == 0)
                {
                  (void) ThrowMagickException(exception,
                     GetMagickModule(),CoderError,
                     "Unknown MAGN method in MNG datastream","`%s'",
                     image->filename);

                   mng_data->magn_warning++;
                }
#ifdef MNG_OBJECT_BUFFERS
          /* Magnify existing objects in the range magn_first to magn_last */
#endif
            if (magn_first == 0 || magn_last == 0)
              {
                /* Save the magnification factors for object 0 */
                mng_data->magn_mb=magn_mb;
                mng_data->magn_ml=magn_ml;
                mng_data->magn_mr=magn_mr;
                mng_data->magn_mt=magn_mt;
                mng_data->magn_mx=magn_mx;
                mng_data->magn_my=magn_my;
                mng_data->magn_methx=magn_methx;
                mng_data->magn_methy=magn_methy;
              }
          }

        if (memcmp(type,mng_PAST,4) == 0)
          {
            if (mng_data->past_warning == 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"PAST is not implemented yet","`%s'",
                image->filename);

            mng_data->past_warning++;
          }

        if (memcmp(type,mng_SHOW,4) == 0)
          {
            if (mng_data->show_warning == 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"SHOW is not implemented yet","`%s'",
                image->filename);

            mng_data->show_warning++;
          }

        if (memcmp(type,mng_sBIT,4) == 0)
          {
            if (length < 4)
              mng_data->have_global_sbit=MagickFalse;

            else
              {
                mng_data->global_sbit.gray=p[0];
                mng_data->global_sbit.red=p[0];
                mng_data->global_sbit.green=p[1];
                mng_data->global_sbit.blue=p[2];
                mng_data->global_sbit.alpha=p[3];
                mng_data->have_global_sbit=MagickTrue;
             }
          }
        if (memcmp(type,mng_pHYs,4) == 0)
          {
            if (length > 8)
              {
                mng_data->global_x_pixels_per_unit=
                    (size_t) mng_get_long(p);
                mng_data->global_y_pixels_per_unit=
                    (size_t) mng_get_long(&p[4]);
                mng_data->global_phys_unit_type=p[8];
                mng_data->have_global_phys=MagickTrue;
              }

            else
              mng_data->have_global_phys=MagickFalse;
          }
        if (memcmp(type,mng_pHYg,4) == 0)
          {
            if (mng_data->phyg_warning == 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"pHYg is not implemented.","`%s'",image->filename);

            mng_data->phyg_warning++;
          }
        if (memcmp(type,mng_BASI,4) == 0)
          {
            skip_to_iend=MagickTrue;

            if (mng_data->basi_warning == 0)
              (void) ThrowMagickException(exception,GetMagickModule(),
                CoderError,"BASI is not implemented yet","`%s'",
                image->filename);

            mng_data->basi_warning++;
#ifdef MNG_BASI_SUPPORTED
            basi_width=(unsigned long) mng_get_long(p);
            basi_width=(unsigned long) mng_get_long(&p[4]);
            basi_color_type=p[8];
            basi_compression_method=p[9];
            basi_filter_type=p[10];
            basi_interlace_method=p[11];
            if (length > 11)
              basi_red=((png_uint_32) p[12] << 8) & (png_uint_32) p[13];

            else
              basi_red=0;

            if (length > 13)
              basi_green=((png_uint_32) p[14] << 8) & (png_uint_32) p[15];

            else
              basi_green=0;

            if (length > 15)
              basi_blue=((png_uint_32) p[16] << 8) & (png_uint_32) p[17];

            else
              basi_blue=0;

            if (length > 17)
              basi_alpha=((png_uint_32) p[18] << 8) & (png_uint_32) p[19];

            else
              {
                if (basi_sample_depth == 16)
                  basi_alpha=65535L;
                else
                  basi_alpha=255;
              }

            if (length > 19)
              basi_viewable=p[20];

            else
              basi_viewable=0;

#endif
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }

        if (memcmp(type,mng_IHDR,4)
#if defined(JNG_SUPPORTED)
            && memcmp(type,mng_JHDR,4)
#endif
            )
          {
            /* Not an IHDR or JHDR chunk */
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);

            continue;
          }
/* Process IHDR */
        if (logging != MagickFalse)
          (void) LogMagickEvent(CoderEvent,GetMagickModule(),
            "  Processing %c%c%c%c chunk",type[0],type[1],type[2],type[3]);

        mng_data->exists[object_id]=MagickTrue;
        mng_data->viewable[object_id]=MagickTrue;

        if (mng_data->invisible[object_id])
          {
            if (logging != MagickFalse)
              (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                "  Skipping invisible object");

            skip_to_iend=MagickTrue;
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            continue;
          }
#if defined(MNG_INSERT_LAYERS)
        if (length < 8)
          {
            chunk=(unsigned char *) RelinquishMagickMemory(chunk);
            ThrowReaderException(CorruptImageError,"ImproperImageHeader");
          }

        image_width=(size_t) mng_get_long(p);
        image_height=(size_t) mng_get_long(&p[4]);
#endif
        chunk=(unsigned char *) RelinquishMagickMemory(chunk);

        /*
          Insert a transparent background layer behind the entire animation
          if it is not full screen.
        */
#if defined(MNG_INSERT_LAYERS)
        if (insert_layers && mng_type && first_mng_object)
          {
            if ((mng_data->clip.left > 0) || (mng_data->clip.top > 0) ||
                (image_width < mng_data->mng_width) ||
                (mng_data->clip.right < (ssize_t) mng_data->mng_width) ||
                (image_height < mng_data->mng_height) ||
                (mng_data->clip.bottom < (ssize_t) mng_data->mng_height))
              {
                if (GetAuthenticPixelQueue(image) != (Quantum *) NULL)
                  {
                    /*
                      Allocate next image structure.
                    */
                    AcquireNextImage(image_info,image,exception);

                    if (GetNextImageInList(image) == (Image *) NULL)
                      return(DestroyImageList(image));

                    image=SyncNextImageInList(image);
                  }
                mng_data->image=image;

                if (term_chunk_found)
                  {
                    image->start_loop=MagickTrue;
                    image->iterations=mng_iterations;
                    term_chunk_found=MagickFalse;
                  }

                else
                    image->start_loop=MagickFalse;

                /* Make a background rectangle.  */

                image->delay=0;
                image->columns=mng_data->mng_width;
                image->rows=mng_data->mng_height;
                image->page.width=mng_data->mng_width;
                image->page.height=mng_data->mng_height;
                image->page.x=0;
                image->page.y=0;
                image->background_color=mng_background_color;
                (void) SetImageBackgroundColor(image,exception);
                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "  Inserted transparent background layer, W=%.20g, H=%.20g",
                    (double) mng_data->mng_width,(double) mng_data->mng_height);
              }
          }
        /*
          Insert a background layer behind the upcoming image if
          framing_mode is 3, and we haven't already inserted one.
        */
        if (insert_layers && (mng_data->framing_mode == 3) &&
                (subframe_width) && (subframe_height) && (simplicity == 0 ||
                (simplicity & 0x08)))
          {
            if (GetAuthenticPixelQueue(image) != (Quantum *) NULL)
            {
              /*
                Allocate next image structure.
              */
              AcquireNextImage(image_info,image,exception);

              if (GetNextImageInList(image) == (Image *) NULL)
                return(DestroyImageList(image));

              image=SyncNextImageInList(image);
            }

            mng_data->image=image;

            if (term_chunk_found)
              {
                image->start_loop=MagickTrue;
                image->iterations=mng_iterations;
                term_chunk_found=MagickFalse;
              }

            else
                image->start_loop=MagickFalse;

            image->delay=0;
            image->columns=subframe_width;
            image->rows=subframe_height;
            image->page.width=subframe_width;
            image->page.height=subframe_height;
            image->page.x=mng_data->clip.left;
            image->page.y=mng_data->clip.top;
            image->background_color=mng_background_color;
            image->alpha_trait=UndefinedPixelTrait;
            (void) SetImageBackgroundColor(image,exception);

            if (logging != MagickFalse)
              (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                "  Insert background layer, L=%.20g, R=%.20g T=%.20g, B=%.20g",
                (double) mng_data->clip.left,(double) mng_data->clip.right,
                (double) mng_data->clip.top,(double) mng_data->clip.bottom);
          }
#endif /* MNG_INSERT_LAYERS */
        first_mng_object=MagickFalse;

        if (GetAuthenticPixelQueue(image) != (Quantum *) NULL)
          {
            /*
              Allocate next image structure.
            */
            AcquireNextImage(image_info,image,exception);

            if (GetNextImageInList(image) == (Image *) NULL)
              return(DestroyImageList(image));

            image=SyncNextImageInList(image);
          }
        mng_data->image=image;
        status=SetImageProgress(image,LoadImagesTag,TellBlob(image),
          GetBlobSize(image));

        if (status == MagickFalse)
          break;

        if (term_chunk_found)
          {
            image->start_loop=MagickTrue;
            term_chunk_found=MagickFalse;
          }

        else
            image->start_loop=MagickFalse;

        if (mng_data->framing_mode == 1 || mng_data->framing_mode == 3)
          {
            image->delay=frame_delay;
            frame_delay=default_frame_delay;
          }

        else
          image->delay=0;

        image->page.width=mng_data->mng_width;
        image->page.height=mng_data->mng_height;
        image->page.x=mng_data->x_off[object_id];
        image->page.y=mng_data->y_off[object_id];
        image->iterations=mng_iterations;

        /*
          Seek back to the beginning of the IHDR or JHDR chunk's length field.
        */

        if (logging != MagickFalse)
          (void) LogMagickEvent(CoderEvent,GetMagickModule(),
            "  Seeking back to beginning of %c%c%c%c chunk",type[0],type[1],
            type[2],type[3]);

        offset=SeekBlob(image,-((ssize_t) length+12),SEEK_CUR);

        if (offset < 0)
          ThrowReaderException(CorruptImageError,"ImproperImageHeader");
      }

    mng_data->image=image;
    mng_data->mng_type=mng_type;
    mng_data->object_id=object_id;

    if (memcmp(type,mng_IHDR,4) == 0)
      image=ReadOnePNGImage(mng_data,image_info,exception);

#if defined(JNG_SUPPORTED)
    else
      image=ReadOneJNGImage(mng_data,image_info,exception);
#endif

    // Check if the image is NULL, log an error message if logging is enabled, and return NULL.
    // If dimensions are zero, close the image blob and return the destroyed image list.
    // If the image is valid with non-zero dimensions, update mng_data to reference this image.
    // <MASK>

    if (mng_type)
      {
        MngBox
          crop_box;

        if (((mng_data->magn_methx > 0) && (mng_data->magn_methx <= 5)) &&
            ((mng_data->magn_methy > 0) && (mng_data->magn_methy <= 5)))
          {
            png_uint_32
               magnified_height,
               magnified_width;

            if (logging != MagickFalse)
              (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                "  Processing MNG MAGN chunk");

            if (image->columns == 1)
              mng_data->magn_methx = 1;
            if (image->rows == 1)
              mng_data->magn_methy = 1;
            if (mng_data->magn_methx == 1)
              {
                magnified_width=mng_data->magn_ml;

                if (image->columns > 1)
                   magnified_width += mng_data->magn_mr;

                if (image->columns > 2)
                   magnified_width += (png_uint_32)
                      ((image->columns-2)*(mng_data->magn_mx));
              }

            else
              {
                magnified_width=(png_uint_32) image->columns;

                if (image->columns > 1)
                   magnified_width += mng_data->magn_ml-1;

                if (image->columns > 2)
                   magnified_width += mng_data->magn_mr-1;

                if (image->columns > 3)
                   magnified_width += (png_uint_32)
                      ((image->columns-3)*(mng_data->magn_mx-1));
              }

            if (mng_data->magn_methy == 1)
              {
                magnified_height=mng_data->magn_mt;

                if (image->rows > 1)
                   magnified_height += mng_data->magn_mb;

                if (image->rows > 2)
                   magnified_height += (png_uint_32)
                      ((image->rows-2)*(mng_data->magn_my));
              }

            else
              {
                magnified_height=(png_uint_32) image->rows;

                if (image->rows > 1)
                   magnified_height += mng_data->magn_mt-1;

                if (image->rows > 2)
                   magnified_height += mng_data->magn_mb-1;

                if (image->rows > 3)
                   magnified_height += (png_uint_32)
                      ((image->rows-3)*(mng_data->magn_my-1));
              }

            if (magnified_height > image->rows ||
                magnified_width > image->columns)
              {
                Image
                  *large_image;

                int
                  yy;

                Quantum
                  *next,
                  *prev;

                png_uint_16
                  magn_methx,
                  magn_methy;

                ssize_t
                  m,
                  y;

                Quantum
                  *n,
                  *q;

                ssize_t
                  x;

                /* Allocate next image structure.  */

                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "    Allocate magnified image");

                AcquireNextImage(image_info,image,exception);

                if (GetNextImageInList(image) == (Image *) NULL)
                  return(DestroyImageList(image));

                large_image=SyncNextImageInList(image);

                large_image->columns=magnified_width;
                large_image->rows=magnified_height;

                magn_methx=mng_data->magn_methx;
                magn_methy=mng_data->magn_methy;

#if (MAGICKCORE_QUANTUM_DEPTH > 16)
#define QM unsigned short
                if (magn_methx != 1 || magn_methy != 1)
                  {
                  /*
                     Scale pixels to unsigned shorts to prevent
                     overflow of intermediate values of interpolations
                  */
                     for (y=0; y < (ssize_t) image->rows; y++)
                     {
                       q=GetAuthenticPixels(image,0,y,image->columns,1,
                          exception);
                       if (q == (Quantum *) NULL)
                         break;
                       for (x=(ssize_t) image->columns-1; x >= 0; x--)
                       {
                          SetPixelRed(image,ScaleQuantumToShort(
                            GetPixelRed(image,q)),q);
                          SetPixelGreen(image,ScaleQuantumToShort(
                            GetPixelGreen(image,q)),q);
                          SetPixelBlue(image,ScaleQuantumToShort(
                            GetPixelBlue(image,q)),q);
                          SetPixelAlpha(image,ScaleQuantumToShort(
                            GetPixelAlpha(image,q)),q);
                          q+=GetPixelChannels(image);
                       }

                       if (SyncAuthenticPixels(image,exception) == MagickFalse)
                         break;
                     }
                  }
#else
#define QM Quantum
#endif

                if (image->alpha_trait != UndefinedPixelTrait)
                   (void) SetImageBackgroundColor(large_image,exception);

                else
                  {
                    large_image->background_color.alpha=OpaqueAlpha;
                    (void) SetImageBackgroundColor(large_image,exception);

                    if (magn_methx == 4)
                      magn_methx=2;

                    if (magn_methx == 5)
                      magn_methx=3;

                    if (magn_methy == 4)
                      magn_methy=2;

                    if (magn_methy == 5)
                      magn_methy=3;
                  }

                /* magnify the rows into the right side of the large image */

                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "    Magnify the rows to %.20g",
                    (double) large_image->rows);
                m=(ssize_t) mng_data->magn_mt;
                yy=0;
                length=(size_t) GetPixelChannels(image)*image->columns;
                next=(Quantum *) AcquireQuantumMemory(length,sizeof(*next));
                prev=(Quantum *) AcquireQuantumMemory(length,sizeof(*prev));

                if ((prev == (Quantum *) NULL) ||
                    (next == (Quantum *) NULL))
                  {
                    if (prev != (Quantum *) NULL)
                      prev=(Quantum *) RelinquishMagickMemory(prev);
                    if (next != (Quantum *) NULL)
                      next=(Quantum *) RelinquishMagickMemory(next);
                    image=DestroyImageList(image);
                    ThrowReaderException(ResourceLimitError,
                      "MemoryAllocationFailed");
                  }

                n=GetAuthenticPixels(image,0,0,image->columns,1,exception);
                (void) memcpy(next,n,length);

                for (y=0; y < (ssize_t) image->rows; y++)
                {
                  if (y == 0)
                    m=(ssize_t) mng_data->magn_mt;

                  else if (magn_methy > 1 && y == (ssize_t) image->rows-2)
                    m=(ssize_t) mng_data->magn_mb;

                  else if (magn_methy <= 1 && y == (ssize_t) image->rows-1)
                    m=(ssize_t) mng_data->magn_mb;

                  else if (magn_methy > 1 && y == (ssize_t) image->rows-1)
                    m=1;

                  else
                    m=(ssize_t) mng_data->magn_my;

                  n=prev;
                  prev=next;
                  next=n;

                  if (y < (ssize_t) image->rows-1)
                    {
                      n=GetAuthenticPixels(image,0,y+1,image->columns,1,
                          exception);
                      (void) memcpy(next,n,length);
                    }

                  for (i=0; i < m; i++, yy++)
                  {
                    Quantum
                      *pixels;

                    assert(yy < (ssize_t) large_image->rows);
                    pixels=prev;
                    n=next;
                    q=GetAuthenticPixels(large_image,0,yy,large_image->columns,
                      1,exception);
                    if (q == (Quantum *) NULL)
                      break;
                    q+=(large_image->columns-image->columns)*
                      GetPixelChannels(large_image);

                    for (x=(ssize_t) image->columns-1; x >= 0; x--)
                    {
                      /* To do: get color as function of indexes[x] */
                      /*
                      if (image->storage_class == PseudoClass)
                        {
                        }
                      */

                      if (magn_methy <= 1)
                        {
                          /* replicate previous */
                          SetPixelRed(large_image,GetPixelRed(image,pixels),q);
                          SetPixelGreen(large_image,GetPixelGreen(image,
                             pixels),q);
                          SetPixelBlue(large_image,GetPixelBlue(image,
                             pixels),q);
                          SetPixelAlpha(large_image,GetPixelAlpha(image,
                             pixels),q);
                        }

                      else if (magn_methy == 2 || magn_methy == 4)
                        {
                          if (i == 0)
                            {
                              SetPixelRed(large_image,GetPixelRed(image,
                                 pixels),q);
                              SetPixelGreen(large_image,GetPixelGreen(image,
                                 pixels),q);
                              SetPixelBlue(large_image,GetPixelBlue(image,
                                 pixels),q);
                              SetPixelAlpha(large_image,GetPixelAlpha(image,
                                 pixels),q);
                            }

                          else
                            {
                              /* Interpolate */
                              SetPixelRed(large_image,((QM) (((ssize_t)
                                 (2*i*(GetPixelRed(image,n)
                                 -GetPixelRed(image,pixels)+m))/
                                 ((ssize_t) (m*2))
                                 +GetPixelRed(image,pixels)))),q);
                              SetPixelGreen(large_image,((QM) (((ssize_t)
                                 (2*i*(GetPixelGreen(image,n)
                                 -GetPixelGreen(image,pixels)+m))/
                                 ((ssize_t) (m*2))
                                 +GetPixelGreen(image,pixels)))),q);
                              SetPixelBlue(large_image,((QM) (((ssize_t)
                                 (2*i*(GetPixelBlue(image,n)
                                 -GetPixelBlue(image,pixels)+m))/
                                 ((ssize_t) (m*2))
                                 +GetPixelBlue(image,pixels)))),q);

                              if (image->alpha_trait != UndefinedPixelTrait)
                                 SetPixelAlpha(large_image, ((QM) (((ssize_t)
                                    (2*i*(GetPixelAlpha(image,n)
                                    -GetPixelAlpha(image,pixels)+m))
                                    /((ssize_t) (m*2))+
                                   GetPixelAlpha(image,pixels)))),q);
                            }

                          if (magn_methy == 4)
                            {
                              /* Replicate nearest */
                              if (i <= ((m+1) << 1))
                                 SetPixelAlpha(large_image,GetPixelAlpha(image,
                                    pixels),q);
                              else
                                 SetPixelAlpha(large_image,GetPixelAlpha(image,
                                    n),q);
                            }
                        }

                      else /* if (magn_methy == 3 || magn_methy == 5) */
                        {
                          /* Replicate nearest */
                          if (i <= ((m+1) << 1))
                          {
                             SetPixelRed(large_image,GetPixelRed(image,
                                    pixels),q);
                             SetPixelGreen(large_image,GetPixelGreen(image,
                                    pixels),q);
                             SetPixelBlue(large_image,GetPixelBlue(image,
                                    pixels),q);
                             SetPixelAlpha(large_image,GetPixelAlpha(image,
                                    pixels),q);
                          }

                          else
                          {
                             SetPixelRed(large_image,GetPixelRed(image,n),q);
                             SetPixelGreen(large_image,GetPixelGreen(image,n),
                                    q);
                             SetPixelBlue(large_image,GetPixelBlue(image,n),
                                    q);
                             SetPixelAlpha(large_image,GetPixelAlpha(image,n),
                                    q);
                          }

                          if (magn_methy == 5)
                            {
                              SetPixelAlpha(large_image,(QM) (((ssize_t) (2*i*
                                 (GetPixelAlpha(image,n)
                                 -GetPixelAlpha(image,pixels))
                                 +m))/((ssize_t) (m*2))
                                 +GetPixelAlpha(image,pixels)),q);
                            }
                        }
                      n+=GetPixelChannels(image);
                      q+=GetPixelChannels(large_image);
                      pixels+=GetPixelChannels(image);
                    } /* x */

                    if (SyncAuthenticPixels(large_image,exception) == 0)
                      break;

                  } /* i */
                } /* y */

                prev=(Quantum *) RelinquishMagickMemory(prev);
                next=(Quantum *) RelinquishMagickMemory(next);

                length=image->columns;

                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "    Delete original image");

                DeleteImageFromList(&image);

                image=large_image;

                mng_data->image=image;

                /* magnify the columns */
                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "    Magnify the columns to %.20g",
                    (double) image->columns);

                for (y=0; y < (ssize_t) image->rows; y++)
                {
                  Quantum
                    *pixels;

                  q=GetAuthenticPixels(image,0,y,image->columns,1,exception);
                  if (q == (Quantum *) NULL)
                    break;
                  pixels=q+(image->columns-length)*GetPixelChannels(image);
                  n=pixels+GetPixelChannels(image);

                  for (x=(ssize_t) (image->columns-length);
                    x < (ssize_t) image->columns; x++)
                  {
                    /* To do: Rewrite using Get/Set***PixelChannel() */

                    if (x == (ssize_t) (image->columns-length))
                      m=(ssize_t) mng_data->magn_ml;

                    else if (magn_methx > 1 && x == (ssize_t) image->columns-2)
                      m=(ssize_t) mng_data->magn_mr;

                    else if (magn_methx <= 1 &&
                        x == (ssize_t) image->columns-1)
                      m=(ssize_t) mng_data->magn_mr;

                    else if (magn_methx > 1 && x == (ssize_t) image->columns-1)
                      m=1;

                    else
                      m=(ssize_t) mng_data->magn_mx;

                    for (i=0; i < m; i++)
                    {
                      if (magn_methx <= 1)
                        {
                          /* replicate previous */
                          SetPixelRed(image,GetPixelRed(image,pixels),q);
                          SetPixelGreen(image,GetPixelGreen(image,pixels),q);
                          SetPixelBlue(image,GetPixelBlue(image,pixels),q);
                          SetPixelAlpha(image,GetPixelAlpha(image,pixels),q);
                        }

                      else if (magn_methx == 2 || magn_methx == 4)
                        {
                          if (i == 0)
                          {
                            SetPixelRed(image,GetPixelRed(image,pixels),q);
                            SetPixelGreen(image,GetPixelGreen(image,pixels),q);
                            SetPixelBlue(image,GetPixelBlue(image,pixels),q);
                            SetPixelAlpha(image,GetPixelAlpha(image,pixels),q);
                          }

                          /* To do: Rewrite using Get/Set***PixelChannel() */
                          else
                            {
                              /* Interpolate */
                              SetPixelRed(image,(QM) ((2*i*(
                                 GetPixelRed(image,n)
                                 -GetPixelRed(image,pixels))+m)
                                 /((ssize_t) (m*2))+
                                 GetPixelRed(image,pixels)),q);

                              SetPixelGreen(image,(QM) ((2*i*(
                                 GetPixelGreen(image,n)
                                 -GetPixelGreen(image,pixels))+m)
                                 /((ssize_t) (m*2))+
                                 GetPixelGreen(image,pixels)),q);

                              SetPixelBlue(image,(QM) ((2*i*(
                                 GetPixelBlue(image,n)
                                 -GetPixelBlue(image,pixels))+m)
                                 /((ssize_t) (m*2))+
                                 GetPixelBlue(image,pixels)),q);
                              if (image->alpha_trait != UndefinedPixelTrait)
                                 SetPixelAlpha(image,(QM) ((2*i*(
                                   GetPixelAlpha(image,n)
                                   -GetPixelAlpha(image,pixels))+m)
                                   /((ssize_t) (m*2))+
                                   GetPixelAlpha(image,pixels)),q);
                            }

                          if (magn_methx == 4)
                            {
                              /* Replicate nearest */
                              if (i <= ((m+1) << 1))
                              {
                                 SetPixelAlpha(image,
                                   GetPixelAlpha(image,pixels)+0,q);
                              }
                              else
                              {
                                 SetPixelAlpha(image,
                                   GetPixelAlpha(image,n)+0,q);
                              }
                            }
                        }

                      else /* if (magn_methx == 3 || magn_methx == 5) */
                        {
                          /* Replicate nearest */
                          if (i <= ((m+1) << 1))
                          {
                             SetPixelRed(image,GetPixelRed(image,pixels),q);
                             SetPixelGreen(image,GetPixelGreen(image,
                                 pixels),q);
                             SetPixelBlue(image,GetPixelBlue(image,pixels),q);
                             SetPixelAlpha(image,GetPixelAlpha(image,
                                 pixels),q);
                          }

                          else
                          {
                             SetPixelRed(image,GetPixelRed(image,n),q);
                             SetPixelGreen(image,GetPixelGreen(image,n),q);
                             SetPixelBlue(image,GetPixelBlue(image,n),q);
                             SetPixelAlpha(image,GetPixelAlpha(image,n),q);
                          }

                          if (magn_methx == 5)
                            {
                              /* Interpolate */
                              SetPixelAlpha(image,
                                 (QM) ((2*i*( GetPixelAlpha(image,n)
                                 -GetPixelAlpha(image,pixels))+m)/
                                 ((ssize_t) (m*2))
                                 +GetPixelAlpha(image,pixels)),q);
                            }
                        }
                      q+=GetPixelChannels(image);
                    }
                    n+=GetPixelChannels(image);
                  }

                  if (SyncAuthenticPixels(image,exception) == MagickFalse)
                    break;
                }
#if (MAGICKCORE_QUANTUM_DEPTH > 16)
              if (magn_methx != 1 || magn_methy != 1)
                {
                /*
                   Rescale pixels to Quantum
                */
                   for (y=0; y < (ssize_t) image->rows; y++)
                   {
                     q=GetAuthenticPixels(image,0,y,image->columns,1,
                       exception);
                     if (q == (Quantum *) NULL)
                       break;

                     for (x=(ssize_t) image->columns-1; x >= 0; x--)
                     {
                        SetPixelRed(image,ScaleShortToQuantum(
                          GetPixelRed(image,q)),q);
                        SetPixelGreen(image,ScaleShortToQuantum(
                          GetPixelGreen(image,q)),q);
                        SetPixelBlue(image,ScaleShortToQuantum(
                          GetPixelBlue(image,q)),q);
                        SetPixelAlpha(image,ScaleShortToQuantum(
                          GetPixelAlpha(image,q)),q);
                        q+=GetPixelChannels(image);
                     }

                     if (SyncAuthenticPixels(image,exception) == MagickFalse)
                       break;
                   }
                }
#endif
                if (logging != MagickFalse)
                  (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                    "  Finished MAGN processing");
              }
          }

        /*
          Crop_box is with respect to the upper left corner of the MNG.
        */
        crop_box.left=mng_data->image_box.left+mng_data->x_off[object_id];
        crop_box.right=mng_data->image_box.right+mng_data->x_off[object_id];
        crop_box.top=mng_data->image_box.top+mng_data->y_off[object_id];
        crop_box.bottom=mng_data->image_box.bottom+mng_data->y_off[object_id];
        crop_box=mng_minimum_box(crop_box,mng_data->clip);
        crop_box=mng_minimum_box(crop_box,mng_data->frame);
        crop_box=mng_minimum_box(crop_box,mng_data->object_clip[object_id]);
        if ((crop_box.left != (mng_data->image_box.left
            +mng_data->x_off[object_id])) ||
            (crop_box.right != (mng_data->image_box.right
            +mng_data->x_off[object_id])) ||
            (crop_box.top != (mng_data->image_box.top
            +mng_data->y_off[object_id])) ||
            (crop_box.bottom != (mng_data->image_box.bottom
            +mng_data->y_off[object_id])))
          {
            if (logging != MagickFalse)
              (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                "  Crop the PNG image");

            if ((crop_box.left < crop_box.right) &&
                (crop_box.top < crop_box.bottom))
              {
                Image
                  *im;

                RectangleInfo
                  crop_info;

                /*
                  Crop_info is with respect to the upper left corner of
                  the image.
                */
                crop_info.x=(crop_box.left-mng_data->x_off[object_id]);
                crop_info.y=(crop_box.top-mng_data->y_off[object_id]);
                crop_info.width=(size_t) (crop_box.right-crop_box.left);
                crop_info.height=(size_t) (crop_box.bottom-crop_box.top);
                image->page.width=image->columns;
                image->page.height=image->rows;
                image->page.x=0;
                image->page.y=0;
                im=CropImage(image,&crop_info,exception);

                if (im != (Image *) NULL)
                  {
                    image->columns=im->columns;
                    image->rows=im->rows;
                    im=DestroyImage(im);
                    image->page.width=image->columns;
                    image->page.height=image->rows;
                    image->page.x=crop_box.left;
                    image->page.y=crop_box.top;
                  }
              }

            else
              {
                /*
                  No pixels in crop area.  The MNG spec still requires
                  a layer, though, so make a single transparent pixel in
                  the top left corner.
                */
                image->columns=1;
                image->rows=1;
                image->colors=2;
                (void) SetImageBackgroundColor(image,exception);
                image->page.width=1;
                image->page.height=1;
                image->page.x=0;
                image->page.y=0;
              }
          }
#ifndef PNG_READ_EMPTY_PLTE_SUPPORTED
        image=mng_data->image;
#endif
      }

#if (MAGICKCORE_QUANTUM_DEPTH > 16)
      /* PNG does not handle depths greater than 16 so reduce it even
       * if lossy.
       */
      if (image->depth > 16)
         image->depth=16;
#endif

#if (MAGICKCORE_QUANTUM_DEPTH > 8)
      if (image->depth > 8)
        {
          /* To do: fill low byte properly */
          image->depth=16;
        }

      if (LosslessReduceDepthOK(image,exception) != MagickFalse)
         image->depth = 8;
#endif

      if (image_info->number_scenes != 0)
        {
          if (mng_data->scenes_found >
             (ssize_t) (image_info->first_scene+image_info->number_scenes))
            break;
        }

      if (logging != MagickFalse)
        (void) LogMagickEvent(CoderEvent,GetMagickModule(),
          "  Finished reading image datastream.");

  } while (LocaleCompare(image_info->magick,"MNG") == 0);

  (void) CloseBlob(image);

  if (logging != MagickFalse)
    (void) LogMagickEvent(CoderEvent,GetMagickModule(),
      "  Finished reading all image datastreams.");

#if defined(MNG_INSERT_LAYERS)
  if (insert_layers && !mng_data->image_found && (mng_data->mng_width) &&
       (mng_data->mng_height))
    {
      /*
        Insert a background layer if nothing else was found.
      */
      if (logging != MagickFalse)
        (void) LogMagickEvent(CoderEvent,GetMagickModule(),
          "  No images found.  Inserting a background layer.");

      if (GetAuthenticPixelQueue(image) != (Quantum *) NULL)
        {
          /*
            Allocate next image structure.
          */
          AcquireNextImage(image_info,image,exception);
          if (GetNextImageInList(image) == (Image *) NULL)
            {
              if (logging != MagickFalse)
                (void) LogMagickEvent(CoderEvent,GetMagickModule(),
                  "  Allocation failed, returning NULL.");

              return(DestroyImageList(image));;
            }
          image=SyncNextImageInList(image);
        }
      image->columns=mng_data->mng_width;
      image->rows=mng_data->mng_height;
      image->page.width=mng_data->mng_width;
      image->page.height=mng_data->mng_height;
      image->page.x=0;
      image->page.y=0;
      image->background_color=mng_background_color;
      image->alpha_trait=UndefinedPixelTrait;

      if (image_info->ping == MagickFalse)
        (void) SetImageBackgroundColor(image,exception);

      mng_data->image_found++;
    }
#endif
  image->iterations=mng_iterations;

  if (mng_iterations == 1)
    image->start_loop=MagickTrue;

  while (GetPreviousImageInList(image) != (Image *) NULL)
  {
    image_count++;
    if (image_count > 10*mng_data->image_found)
      {
        if (logging != MagickFalse)
          (void) LogMagickEvent(CoderEvent,GetMagickModule(),"  No beginning");

        (void) ThrowMagickException(exception,GetMagickModule(),
          CoderError,"Linked list is corrupted, beginning of list not found",
          "`%s'",image_info->filename);

        return(DestroyImageList(image));
      }

    image=GetPreviousImageInList(image);

    if (GetNextImageInList(image) == (Image *) NULL)
      {
        if (logging != MagickFalse)
          (void) LogMagickEvent(CoderEvent,GetMagickModule(),"  Corrupt list");

        (void) ThrowMagickException(exception,GetMagickModule(),
          CoderError,"Linked list is corrupted; next_image is NULL","`%s'",
          image_info->filename);
      }
  }

  if (mng_data->ticks_per_second && mng_data->image_found > 1 &&
             GetNextImageInList(image) ==
     (Image *) NULL)
    {
      if (logging != MagickFalse)
        (void) LogMagickEvent(CoderEvent,GetMagickModule(),
            "  First image null");

      (void) ThrowMagickException(exception,GetMagickModule(),
        CoderError,"image->next for first image is NULL but shouldn't be.",
        "`%s'",image_info->filename);
    }

  if (mng_data->image_found == 0)
    {
      if (logging != MagickFalse)
        (void) LogMagickEvent(CoderEvent,GetMagickModule(),
          "  No visible images found.");

      (void) ThrowMagickException(exception,GetMagickModule(),
        CoderError,"No visible images in file","`%s'",image_info->filename);

      return(DestroyImageList(image));
    }

  if (mng_data->ticks_per_second)
    final_delay=1UL*MagickMax(image->ticks_per_second,1L)*
            final_delay/mng_data->ticks_per_second;

  else
    image->start_loop=MagickTrue;

  /* Find final nonzero image delay */
  final_image_delay=0;

  while (GetNextImageInList(image) != (Image *) NULL)
    {
      if (image->delay)
        final_image_delay=image->delay;

      image=GetNextImageInList(image);
    }

  if (final_delay < final_image_delay)
    final_delay=final_image_delay;

  image->delay=final_delay;

  if (logging != MagickFalse)
      (void) LogMagickEvent(CoderEvent,GetMagickModule(),
        "  image->delay=%.20g, final_delay=%.20g",(double) image->delay,
        (double) final_delay);

  if (logging != MagickFalse)
    {
      int
        scene;

      scene=0;
      image=GetFirstImageInList(image);

      (void) LogMagickEvent(CoderEvent,GetMagickModule(),
        "  Before coalesce:");

      (void) LogMagickEvent(CoderEvent,GetMagickModule(),
        "    scene 0 delay=%.20g",(double) image->delay);

      while (GetNextImageInList(image) != (Image *) NULL)
      {
        image=GetNextImageInList(image);
        (void) LogMagickEvent(CoderEvent,GetMagickModule(),
          "    scene %.20g delay=%.20g",(double) scene++,
          (double) image->delay);
      }
    }

  image=GetFirstImageInList(image);
#ifdef MNG_COALESCE_LAYERS
  if (insert_layers && image->next)
    {
      Image
        *next_image,
        *next;

      size_t
        scene;

      if (logging != MagickFalse)
        (void) LogMagickEvent(CoderEvent,GetMagickModule(),
          "  Coalesce Images");

      scene=image->scene;
      next_image=CoalesceImages(image,exception);
      image=DestroyImageList(image);
      if (next_image == (Image *) NULL)
        return((Image *) NULL);
      image=next_image;

      for (next=image; next != (Image *) NULL; next=next_image)
      {
         next->page.width=mng_data->mng_width;
         next->page.height=mng_data->mng_height;
         next->page.x=0;
         next->page.y=0;
         next->scene=scene++;
         next_image=GetNextImageInList(next);

         if (next_image == (Image *) NULL)
           break;

         if (next->delay == 0)
           {
             scene--;
             next_image->previous=GetPreviousImageInList(next);
             if (GetPreviousImageInList(next) == (Image *) NULL)
               image=next_image;
             else
               next->previous->next=next_image;
             next=DestroyImage(next);
           }
      }
    }
#endif

  while (GetNextImageInList(image) != (Image *) NULL)
      image=GetNextImageInList(image);

  image->dispose=BackgroundDispose;

  if (logging != MagickFalse)
    {
      int
        scene;

      scene=0;
      image=GetFirstImageInList(image);

      (void) LogMagickEvent(CoderEvent,GetMagickModule(),
        "  After coalesce:");

      (void) LogMagickEvent(CoderEvent,GetMagickModule(),
        "    scene 0 delay=%.20g dispose=%.20g",(double) image->delay,
        (double) image->dispose);

      while (GetNextImageInList(image) != (Image *) NULL)
      {
        image=GetNextImageInList(image);

        (void) LogMagickEvent(CoderEvent,GetMagickModule(),
          "    scene %.20g delay=%.20g dispose=%.20g",(double) scene++,
          (double) image->delay,(double) image->dispose);
      }
   }

  if (logging != MagickFalse)
    (void) LogMagickEvent(CoderEvent,GetMagickModule(),
      "  exit ReadOneMNGImage();");

  return(image);
}
