static MagickBooleanType RenderMVGContent(Image *image,
  const DrawInfo *draw_info,const size_t depth,ExceptionInfo *exception)
{
#define RenderImageTag  "Render/Image"

  AffineMatrix
    affine,
    current;

  char
    keyword[MagickPathExtent],
    geometry[MagickPathExtent],
    *next_token,
    pattern[MagickPathExtent],
    *primitive,
    *token;

  const char
    *q;

  double
    angle,
    coordinates,
    cursor,
    factor,
    primitive_extent;

  DrawInfo
    *clone_info,
    **graphic_context;

  MagickBooleanType
    proceed;

  MagickStatusType
    status;

  MVGInfo
    mvg_info;

  PointInfo
    point;

  PrimitiveInfo
    *primitive_info;

  PrimitiveType
    primitive_type;

  register const char
    *p;

  register ssize_t
    i,
    x;

  SegmentInfo
    bounds;

  size_t
    extent,
    number_points,
    number_stops;

  SplayTreeInfo
    *macros;

  ssize_t
    defsDepth,
    j,
    k,
    graphiccontextindex,
    symbolDepth;

  StopInfo
    *stops;

  TypeMetric
    metrics;

  assert(image != (Image *) NULL);
  assert(image->signature == MagickCoreSignature);
  if (image->debug != MagickFalse)
    (void) LogMagickEvent(TraceEvent,GetMagickModule(),"%s",image->filename);
  assert(draw_info != (DrawInfo *) NULL);
  assert(draw_info->signature == MagickCoreSignature);
  if (image->debug != MagickFalse)
    (void) LogMagickEvent(TraceEvent,GetMagickModule(),"...");
  if (depth > MagickMaxRecursionDepth)
    ThrowBinaryException(DrawError,"VectorGraphicsNestedTooDeeply",
      image->filename);
  if ((draw_info->primitive == (char *) NULL) ||
      (*draw_info->primitive == '\0'))
    return(MagickFalse);
  if (image->debug != MagickFalse)
    (void) LogMagickEvent(DrawEvent,GetMagickModule(),"begin draw-image");
  if (SetImageStorageClass(image,DirectClass,exception) == MagickFalse)
    return(MagickFalse);
  if (image->alpha_trait == UndefinedPixelTrait)
    {
      status=SetImageAlphaChannel(image,OpaqueAlphaChannel,exception);
      if (status == MagickFalse)
        return(MagickFalse);
    }
  if ((*draw_info->primitive == '@') && (strlen(draw_info->primitive) > 1) &&
      (*(draw_info->primitive+1) != '-') && (depth == 0))
    primitive=FileToString(draw_info->primitive+1,~0UL,exception);
  else
    primitive=AcquireString(draw_info->primitive);
  if (primitive == (char *) NULL)
    return(MagickFalse);
  primitive_extent=(double) strlen(primitive);
  (void) SetImageArtifact(image,"mvg:vector-graphics",primitive);
  graphiccontextindex=0;
  number_stops=0;
  stops=(StopInfo *) NULL;
  /*
    Allocate primitive info memory.
  */
  graphic_context=(DrawInfo **) AcquireMagickMemory(sizeof(*graphic_context));
  if (graphic_context == (DrawInfo **) NULL)
    {
      primitive=DestroyString(primitive);
      ThrowBinaryException(ResourceLimitError,"MemoryAllocationFailed",
        image->filename);
    }
  number_points=PrimitiveExtentPad;
  primitive_info=(PrimitiveInfo *) AcquireQuantumMemory((size_t) number_points,
    sizeof(*primitive_info));
  if (primitive_info == (PrimitiveInfo *) NULL)
    {
      primitive=DestroyString(primitive);
      for ( ; graphiccontextindex >= 0; graphiccontextindex--)
        graphic_context[graphiccontextindex]=DestroyDrawInfo(graphic_context[graphiccontextindex]);
      graphic_context=(DrawInfo **) RelinquishMagickMemory(graphic_context);
      ThrowBinaryException(ResourceLimitError,"MemoryAllocationFailed",
        image->filename);
    }
  (void) memset(primitive_info,0,(size_t) number_points*
    sizeof(*primitive_info));
  (void) memset(&mvg_info,0,sizeof(mvg_info));
  mvg_info.primitive_info=(&primitive_info);
  mvg_info.extent=(&number_points);
  mvg_info.exception=exception;
  graphic_context[graphiccontextindex]=CloneDrawInfo((ImageInfo *) NULL,draw_info);
  graphic_context[graphiccontextindex]->viewbox=image->page;
  if ((image->page.width == 0) || (image->page.height == 0))
    {
      graphic_context[graphiccontextindex]->viewbox.width=image->columns;
      graphic_context[graphiccontextindex]->viewbox.height=image->rows;
    }
  token=AcquireString(primitive);
  extent=strlen(token)+MagickPathExtent;
  defsDepth=0;
  symbolDepth=0;
  cursor=0.0;
  macros=GetMVGMacros(primitive);
  status=MagickTrue;
  for (q=primitive; *q != '\0'; )
  {
    /*
      Interpret graphic primitive.
    */
    if (GetNextToken(q,&q,MagickPathExtent,keyword) < 1)
      break;
    if (*keyword == '\0')
      break;
    if (*keyword == '#')
      {
        /*
          Comment.
        */
        while ((*q != '\n') && (*q != '\0'))
          q++;
        continue;
      }
    p=q-strlen(keyword)-1;
    primitive_type=UndefinedPrimitive;
    current=graphic_context[graphiccontextindex]->affine;
    GetAffineMatrix(&affine);
    *token='\0';
    switch (*keyword)
    {
      case ';':
        break;
      case 'a':
      case 'A':
      {
        if (LocaleCompare("affine",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            affine.sx=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            affine.rx=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            affine.ry=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            affine.sy=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            affine.tx=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            affine.ty=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        if (LocaleCompare("alpha",keyword) == 0)
          {
            primitive_type=AlphaPrimitive;
            break;
          }
        if (LocaleCompare("arc",keyword) == 0)
          {
            primitive_type=ArcPrimitive;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'b':
      case 'B':
      {
        if (LocaleCompare("bezier",keyword) == 0)
          {
            primitive_type=BezierPrimitive;
            break;
          }
        if (LocaleCompare("border-color",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            status&=QueryColorCompliance(token,AllCompliance,
              &graphic_context[graphiccontextindex]->border_color,exception);
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'c':
      case 'C':
      {
        if (LocaleCompare("class",keyword) == 0)
          {
            const char
              *mvg_class;

            (void) GetNextToken(q,&q,extent,token);
            if (*token == '\0')
              {
                status=MagickFalse;
                break;
              }
            // Check if the current token matches the graphic context ID, break if true.
            // Retrieve the MVG class definition from the macros using the token.
            // If the class definition is found,
            // inject the class elements into the stream.
            // Concatenate the MVG class to the elements string, followed by the remaining content.
            // Free the previous primitive string and update it with the new elements.
            // Adjust the stream pointer to the new position in the modified primitive.
            // <MASK>
            break;
          }
        if (LocaleCompare("clip-path",keyword) == 0)
          {
            const char
              *clip_path;

            /*
              Take a node from within the MVG document, and duplicate it here.
            */
            (void) GetNextToken(q,&q,extent,token);
            if (*token == '\0')
              {
                status=MagickFalse;
                break;
              }
            (void) CloneString(&graphic_context[graphiccontextindex]->clip_mask,token);
            clip_path=(const char *) GetValueFromSplayTree(macros,token);
            if (clip_path != (const char *) NULL)
              {
                if (graphic_context[graphiccontextindex]->clipping_mask != (Image *) NULL)
                  graphic_context[graphiccontextindex]->clipping_mask=
                    DestroyImage(graphic_context[graphiccontextindex]->clipping_mask);
                graphic_context[graphiccontextindex]->clipping_mask=DrawClippingMask(image,
                  graphic_context[graphiccontextindex],token,clip_path,exception);
                if (graphic_context[graphiccontextindex]->compliance != SVGCompliance)
                  {
                    clip_path=(const char *) GetValueFromSplayTree(macros,
                      graphic_context[graphiccontextindex]->clip_mask);
                    if (clip_path != (const char *) NULL)
                      (void) SetImageArtifact(image,
                        graphic_context[graphiccontextindex]->clip_mask,clip_path);
                    status&=DrawClipPath(image,graphic_context[graphiccontextindex],
                      graphic_context[graphiccontextindex]->clip_mask,exception);
                  }
              }
            break;
          }
        if (LocaleCompare("clip-rule",keyword) == 0)
          {
            ssize_t
              fill_rule;

            (void) GetNextToken(q,&q,extent,token);
            fill_rule=ParseCommandOption(MagickFillRuleOptions,MagickFalse,
              token);
            if (fill_rule == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->fill_rule=(FillRule) fill_rule;
            break;
          }
        if (LocaleCompare("clip-units",keyword) == 0)
          {
            ssize_t
              clip_units;

            (void) GetNextToken(q,&q,extent,token);
            clip_units=ParseCommandOption(MagickClipPathOptions,MagickFalse,
              token);
            if (clip_units == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->clip_units=(ClipPathUnits) clip_units;
            if (clip_units == ObjectBoundingBox)
              {
                GetAffineMatrix(&current);
                affine.sx=draw_info->bounds.x2;
                affine.sy=draw_info->bounds.y2;
                affine.tx=draw_info->bounds.x1;
                affine.ty=draw_info->bounds.y1;
                break;
              }
            break;
          }
        if (LocaleCompare("circle",keyword) == 0)
          {
            primitive_type=CirclePrimitive;
            break;
          }
        if (LocaleCompare("color",keyword) == 0)
          {
            primitive_type=ColorPrimitive;
            break;
          }
        if (LocaleCompare("compliance",keyword) == 0)
          {
            /*
              MVG compliance associates a clipping mask with an image; SVG
              compliance associates a clipping mask with a graphics context.
            */
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->compliance=(ComplianceType) ParseCommandOption(
              MagickComplianceOptions,MagickFalse,token);
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'd':
      case 'D':
      {
        if (LocaleCompare("decorate",keyword) == 0)
          {
            ssize_t
              decorate;

            (void) GetNextToken(q,&q,extent,token);
            decorate=ParseCommandOption(MagickDecorateOptions,MagickFalse,
              token);
            if (decorate == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->decorate=(DecorationType) decorate;
            break;
          }
        if (LocaleCompare("density",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            (void) CloneString(&graphic_context[graphiccontextindex]->density,token);
            break;
          }
        if (LocaleCompare("direction",keyword) == 0)
          {
            ssize_t
              direction;

            (void) GetNextToken(q,&q,extent,token);
            direction=ParseCommandOption(MagickDirectionOptions,MagickFalse,
              token);
            if (direction == -1)
              status=MagickFalse;
            else
              graphic_context[graphiccontextindex]->direction=(DirectionType) direction;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'e':
      case 'E':
      {
        if (LocaleCompare("ellipse",keyword) == 0)
          {
            primitive_type=EllipsePrimitive;
            break;
          }
        if (LocaleCompare("encoding",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            (void) CloneString(&graphic_context[graphiccontextindex]->encoding,token);
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'f':
      case 'F':
      {
        if (LocaleCompare("fill",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            if (graphic_context[graphiccontextindex]->clip_path != MagickFalse)
              break;
            (void) FormatLocaleString(pattern,MagickPathExtent,"%s",token);
            if (GetImageArtifact(image,pattern) != (const char *) NULL)
              (void) DrawPatternPath(image,draw_info,token,
                &graphic_context[graphiccontextindex]->fill_pattern,exception);
            else
              {
                status&=QueryColorCompliance(token,AllCompliance,
                  &graphic_context[graphiccontextindex]->fill,exception);
                if (graphic_context[graphiccontextindex]->fill_alpha != OpaqueAlpha)
                  graphic_context[graphiccontextindex]->fill.alpha=graphic_context[graphiccontextindex]->fill_alpha;
              }
            break;
          }
        if (LocaleCompare("fill-opacity",keyword) == 0)
          {
            double
              opacity;

            (void) GetNextToken(q,&q,extent,token);
            if (graphic_context[graphiccontextindex]->clip_path != MagickFalse)
              break;
            factor=strchr(token,'%') != (char *) NULL ? 0.01 : 1.0;
            opacity=MagickMin(MagickMax(factor*
              GetDrawValue(token,&next_token),0.0),1.0);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            if (graphic_context[graphiccontextindex]->compliance == SVGCompliance)
              graphic_context[graphiccontextindex]->fill_alpha*=opacity;
            else
              graphic_context[graphiccontextindex]->fill_alpha=QuantumRange*opacity;
            if (graphic_context[graphiccontextindex]->fill.alpha != TransparentAlpha)
              graphic_context[graphiccontextindex]->fill.alpha=graphic_context[graphiccontextindex]->fill_alpha;
            else
              graphic_context[graphiccontextindex]->fill.alpha=(MagickRealType)
                ClampToQuantum(QuantumRange*(1.0-opacity));
            break;
          }
        if (LocaleCompare("fill-rule",keyword) == 0)
          {
            ssize_t
              fill_rule;

            (void) GetNextToken(q,&q,extent,token);
            fill_rule=ParseCommandOption(MagickFillRuleOptions,MagickFalse,
              token);
            if (fill_rule == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->fill_rule=(FillRule) fill_rule;
            break;
          }
        if (LocaleCompare("font",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            (void) CloneString(&graphic_context[graphiccontextindex]->font,token);
            if (LocaleCompare("none",token) == 0)
              graphic_context[graphiccontextindex]->font=(char *) RelinquishMagickMemory(
                graphic_context[graphiccontextindex]->font);
            break;
          }
        if (LocaleCompare("font-family",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            (void) CloneString(&graphic_context[graphiccontextindex]->family,token);
            break;
          }
        if (LocaleCompare("font-size",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->pointsize=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        if (LocaleCompare("font-stretch",keyword) == 0)
          {
            ssize_t
              stretch;

            (void) GetNextToken(q,&q,extent,token);
            stretch=ParseCommandOption(MagickStretchOptions,MagickFalse,token);
            if (stretch == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->stretch=(StretchType) stretch;
            break;
          }
        if (LocaleCompare("font-style",keyword) == 0)
          {
            ssize_t
              style;

            (void) GetNextToken(q,&q,extent,token);
            style=ParseCommandOption(MagickStyleOptions,MagickFalse,token);
            if (style == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->style=(StyleType) style;
            break;
          }
        if (LocaleCompare("font-weight",keyword) == 0)
          {
            ssize_t
              weight;

            (void) GetNextToken(q,&q,extent,token);
            weight=ParseCommandOption(MagickWeightOptions,MagickFalse,token);
            if (weight == -1)
              weight=(ssize_t) StringToUnsignedLong(token);
            graphic_context[graphiccontextindex]->weight=(size_t) weight;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'g':
      case 'G':
      {
        if (LocaleCompare("gradient-units",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            break;
          }
        if (LocaleCompare("gravity",keyword) == 0)
          {
            ssize_t
              gravity;

            (void) GetNextToken(q,&q,extent,token);
            gravity=ParseCommandOption(MagickGravityOptions,MagickFalse,token);
            if (gravity == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->gravity=(GravityType) gravity;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'i':
      case 'I':
      {
        if (LocaleCompare("image",keyword) == 0)
          {
            ssize_t
              compose;

            primitive_type=ImagePrimitive;
            (void) GetNextToken(q,&q,extent,token);
            compose=ParseCommandOption(MagickComposeOptions,MagickFalse,token);
            if (compose == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->compose=(CompositeOperator) compose;
            break;
          }
        if (LocaleCompare("interline-spacing",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->interline_spacing=GetDrawValue(token,
              &next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        if (LocaleCompare("interword-spacing",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->interword_spacing=GetDrawValue(token,
              &next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'k':
      case 'K':
      {
        if (LocaleCompare("kerning",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->kerning=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'l':
      case 'L':
      {
        if (LocaleCompare("letter-spacing",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            if (IsPoint(token) == MagickFalse)
              break;
            clone_info=CloneDrawInfo((ImageInfo *) NULL,graphic_context[graphiccontextindex]);
            clone_info->text=AcquireString(" ");
            status&=GetTypeMetrics(image,clone_info,&metrics,exception);
            graphic_context[graphiccontextindex]->kerning=metrics.width*
              GetDrawValue(token,&next_token);
            clone_info=DestroyDrawInfo(clone_info);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        if (LocaleCompare("line",keyword) == 0)
          {
            primitive_type=LinePrimitive;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'm':
      case 'M':
      {
        if (LocaleCompare("mask",keyword) == 0)
          {
            const char
              *mask_path;

            /*
              Take a node from within the MVG document, and duplicate it here.
            */
            (void) GetNextToken(q,&q,extent,token);
            mask_path=(const char *) GetValueFromSplayTree(macros,token);
            if (mask_path != (const char *) NULL)
              {
                if (graphic_context[graphiccontextindex]->composite_mask != (Image *) NULL)
                  graphic_context[graphiccontextindex]->composite_mask=
                    DestroyImage(graphic_context[graphiccontextindex]->composite_mask);
                graphic_context[graphiccontextindex]->composite_mask=DrawCompositeMask(image,
                  graphic_context[graphiccontextindex],token,mask_path,exception);
                if (graphic_context[graphiccontextindex]->compliance != SVGCompliance)
                  status=SetImageMask(image,CompositePixelMask,
                    graphic_context[graphiccontextindex]->composite_mask,exception);
              }
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'o':
      case 'O':
      {
        if (LocaleCompare("offset",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            break;
          }
        if (LocaleCompare("opacity",keyword) == 0)
          {
            double
              opacity;

            (void) GetNextToken(q,&q,extent,token);
            if (graphic_context[graphiccontextindex]->clip_path != MagickFalse)
              break;
            factor=strchr(token,'%') != (char *) NULL ? 0.01 : 1.0;
            opacity=MagickMin(MagickMax(factor*
              GetDrawValue(token,&next_token),0.0),1.0);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            if (graphic_context[graphiccontextindex]->compliance == SVGCompliance)
              {
                graphic_context[graphiccontextindex]->fill_alpha*=opacity;
                graphic_context[graphiccontextindex]->stroke_alpha*=opacity;
              }
            else
              {
                graphic_context[graphiccontextindex]->fill_alpha=QuantumRange*opacity;
                graphic_context[graphiccontextindex]->stroke_alpha=QuantumRange*opacity;
              }
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'p':
      case 'P':
      {
        if (LocaleCompare("path",keyword) == 0)
          {
            primitive_type=PathPrimitive;
            break;
          }
        if (LocaleCompare("point",keyword) == 0)
          {
            primitive_type=PointPrimitive;
            break;
          }
        if (LocaleCompare("polyline",keyword) == 0)
          {
            primitive_type=PolylinePrimitive;
            break;
          }
        if (LocaleCompare("polygon",keyword) == 0)
          {
            primitive_type=PolygonPrimitive;
            break;
          }
        if (LocaleCompare("pop",keyword) == 0)
          {
            if (GetNextToken(q,&q,extent,token) < 1)
              break;
            if (LocaleCompare("class",token) == 0)
              break;
            if (LocaleCompare("clip-path",token) == 0)
              break;
            if (LocaleCompare("defs",token) == 0)
              {
                defsDepth--;
                graphic_context[graphiccontextindex]->render=defsDepth > 0 ? MagickFalse :
                  MagickTrue;
                break;
              }
            if (LocaleCompare("gradient",token) == 0)
              break;
            if (LocaleCompare("graphic-context",token) == 0)
              {
                if (graphiccontextindex <= 0)
                  {
                    (void) ThrowMagickException(exception,GetMagickModule(),
                      DrawError,"UnbalancedGraphicContextPushPop","`%s'",token);
                    status=MagickFalse;
                    graphiccontextindex=0;
                    break;
                  }
                if ((graphic_context[graphiccontextindex]->clip_mask != (char *) NULL) &&
                    (graphic_context[graphiccontextindex]->compliance != SVGCompliance))
                  if (LocaleCompare(graphic_context[graphiccontextindex]->clip_mask,
                      graphic_context[graphiccontextindex-1]->clip_mask) != 0)
                    status=SetImageMask(image,WritePixelMask,(Image *) NULL,
                      exception);
                graphic_context[graphiccontextindex]=DestroyDrawInfo(graphic_context[graphiccontextindex]);
                graphiccontextindex--;
                break;
              }
            if (LocaleCompare("mask",token) == 0)
              break;
            if (LocaleCompare("pattern",token) == 0)
              break;
            if (LocaleCompare("symbol",token) == 0)
              {
                symbolDepth--;
                graphic_context[graphiccontextindex]->render=symbolDepth > 0 ? MagickFalse :
                  MagickTrue;
                break;
              }
            status=MagickFalse;
            break;
          }
        if (LocaleCompare("push",keyword) == 0)
          {
            if (GetNextToken(q,&q,extent,token) < 1)
              break;
            if (LocaleCompare("class",token) == 0)
              {
                /*
                  Class context.
                */
                for (p=q; *q != '\0'; )
                {
                  if (GetNextToken(q,&q,extent,token) < 1)
                    break;
                  if (LocaleCompare(token,"pop") != 0)
                    continue;
                  (void) GetNextToken(q,(const char **) NULL,extent,token);
                  if (LocaleCompare(token,"class") != 0)
                    continue;
                  break;
                }
                (void) GetNextToken(q,&q,extent,token);
                break;
              }
            if (LocaleCompare("clip-path",token) == 0)
              {
                (void) GetNextToken(q,&q,extent,token);
                for (p=q; *q != '\0'; )
                {
                	if (GetNextToken(q,&q,extent,token) < 1)
                    break;
                	if (LocaleCompare(token,"pop") != 0)
                		continue;
                	(void) GetNextToken(q,(const char **) NULL,extent,token);
                	if (LocaleCompare(token,"clip-path") != 0)
                		continue;
                	break;
                }
                if ((q == (char *) NULL) || (p == (char *) NULL) || ((q-4) < p))
                  {
                    status=MagickFalse;
                    break;
                  }
                (void) GetNextToken(q,&q,extent,token);
                break;
              }
            if (LocaleCompare("defs",token) == 0)
              {
                defsDepth++;
                graphic_context[graphiccontextindex]->render=defsDepth > 0 ? MagickFalse :
                  MagickTrue;
                break;
              }
            if (LocaleCompare("gradient",token) == 0)
              {
                char
                  key[2*MagickPathExtent],
                  name[MagickPathExtent],
                  type[MagickPathExtent];

                SegmentInfo
                  segment;

                (void) GetNextToken(q,&q,extent,token);
                (void) CopyMagickString(name,token,MagickPathExtent);
                (void) GetNextToken(q,&q,extent,token);
                (void) CopyMagickString(type,token,MagickPathExtent);
                (void) GetNextToken(q,&q,extent,token);
                segment.x1=GetDrawValue(token,&next_token);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                (void) GetNextToken(q,&q,extent,token);
                if (*token == ',')
                  (void) GetNextToken(q,&q,extent,token);
                segment.y1=GetDrawValue(token,&next_token);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                (void) GetNextToken(q,&q,extent,token);
                if (*token == ',')
                  (void) GetNextToken(q,&q,extent,token);
                segment.x2=GetDrawValue(token,&next_token);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                (void) GetNextToken(q,&q,extent,token);
                if (*token == ',')
                  (void) GetNextToken(q,&q,extent,token);
                segment.y2=GetDrawValue(token,&next_token);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                if (LocaleCompare(type,"radial") == 0)
                  {
                    (void) GetNextToken(q,&q,extent,token);
                    if (*token == ',')
                      (void) GetNextToken(q,&q,extent,token);
                  }
                for (p=q; *q != '\0'; )
                {
                  if (GetNextToken(q,&q,extent,token) < 1)
                    break;
                  if (LocaleCompare(token,"pop") != 0)
                    continue;
                  (void) GetNextToken(q,(const char **) NULL,extent,token);
                  if (LocaleCompare(token,"gradient") != 0)
                    continue;
                  break;
                }
                if ((q == (char *) NULL) || (p == (char *) NULL) || ((q-4) < p))
                  {
                    status=MagickFalse;
                    break;
                  }
                (void) CopyMagickString(token,p,(size_t) (q-p-4+1));
                bounds.x1=graphic_context[graphiccontextindex]->affine.sx*segment.x1+
                  graphic_context[graphiccontextindex]->affine.ry*segment.y1+
                  graphic_context[graphiccontextindex]->affine.tx;
                bounds.y1=graphic_context[graphiccontextindex]->affine.rx*segment.x1+
                  graphic_context[graphiccontextindex]->affine.sy*segment.y1+
                  graphic_context[graphiccontextindex]->affine.ty;
                bounds.x2=graphic_context[graphiccontextindex]->affine.sx*segment.x2+
                  graphic_context[graphiccontextindex]->affine.ry*segment.y2+
                  graphic_context[graphiccontextindex]->affine.tx;
                bounds.y2=graphic_context[graphiccontextindex]->affine.rx*segment.x2+
                  graphic_context[graphiccontextindex]->affine.sy*segment.y2+
                  graphic_context[graphiccontextindex]->affine.ty;
                (void) FormatLocaleString(key,MagickPathExtent,"%s",name);
                (void) SetImageArtifact(image,key,token);
                (void) FormatLocaleString(key,MagickPathExtent,"%s-type",name);
                (void) SetImageArtifact(image,key,type);
                (void) FormatLocaleString(key,MagickPathExtent,"%s-geometry",
                  name);
                (void) FormatLocaleString(geometry,MagickPathExtent,
                  "%gx%g%+.15g%+.15g",
                  MagickMax(fabs(bounds.x2-bounds.x1+1.0),1.0),
                  MagickMax(fabs(bounds.y2-bounds.y1+1.0),1.0),
                  bounds.x1,bounds.y1);
                (void) SetImageArtifact(image,key,geometry);
                (void) GetNextToken(q,&q,extent,token);
                break;
              }
            if (LocaleCompare("graphic-context",token) == 0)
              {
                graphiccontextindex++;
                graphic_context=(DrawInfo **) ResizeQuantumMemory(
                  graphic_context,(size_t) (graphiccontextindex+1),sizeof(*graphic_context));
                if (graphic_context == (DrawInfo **) NULL)
                  {
                    (void) ThrowMagickException(exception,GetMagickModule(),
                      ResourceLimitError,"MemoryAllocationFailed","`%s'",
                      image->filename);
                    break;
                  }
                graphic_context[graphiccontextindex]=CloneDrawInfo((ImageInfo *) NULL,
                  graphic_context[graphiccontextindex-1]);
                if (*q == '"')
                  {
                    (void) GetNextToken(q,&q,extent,token);
                    (void) CloneString(&graphic_context[graphiccontextindex]->id,token);
                  }
                break;
              }
            if (LocaleCompare("mask",token) == 0)
              {
                (void) GetNextToken(q,&q,extent,token);
                break;
              }
            if (LocaleCompare("pattern",token) == 0)
              {
                char
                  key[2*MagickPathExtent],
                  name[MagickPathExtent];

                RectangleInfo
                  bounds;

                (void) GetNextToken(q,&q,extent,token);
                (void) CopyMagickString(name,token,MagickPathExtent);
                (void) GetNextToken(q,&q,extent,token);
                bounds.x=(ssize_t) ceil(GetDrawValue(token,&next_token)-0.5);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                (void) GetNextToken(q,&q,extent,token);
                if (*token == ',')
                  (void) GetNextToken(q,&q,extent,token);
                bounds.y=(ssize_t) ceil(GetDrawValue(token,&next_token)-0.5);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                (void) GetNextToken(q,&q,extent,token);
                if (*token == ',')
                  (void) GetNextToken(q,&q,extent,token);
                bounds.width=(size_t) floor(GetDrawValue(token,&next_token)+
                  0.5);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                (void) GetNextToken(q,&q,extent,token);
                if (*token == ',')
                  (void) GetNextToken(q,&q,extent,token);
                bounds.height=(size_t) floor(GetDrawValue(token,&next_token)+
                  0.5);
                if (token == next_token)
                  ThrowPointExpectedException(token,exception);
                for (p=q; *q != '\0'; )
                {
                  if (GetNextToken(q,&q,extent,token) < 1)
                    break;
                  if (LocaleCompare(token,"pop") != 0)
                    continue;
                  (void) GetNextToken(q,(const char **) NULL,extent,token);
                  if (LocaleCompare(token,"pattern") != 0)
                    continue;
                  break;
                }
                if ((q == (char *) NULL) || (p == (char *) NULL) || ((q-4) < p))
                  {
                    status=MagickFalse;
                    break;
                  }
                (void) CopyMagickString(token,p,(size_t) (q-p-4+1));
                (void) FormatLocaleString(key,MagickPathExtent,"%s",name);
                (void) SetImageArtifact(image,key,token);
                (void) FormatLocaleString(key,MagickPathExtent,"%s-geometry",
                  name);
                (void) FormatLocaleString(geometry,MagickPathExtent,
                  "%.20gx%.20g%+.20g%+.20g",(double) bounds.width,(double)
                  bounds.height,(double) bounds.x,(double) bounds.y);
                (void) SetImageArtifact(image,key,geometry);
                (void) GetNextToken(q,&q,extent,token);
                break;
              }
            if (LocaleCompare("symbol",token) == 0)
              {
                symbolDepth++;
                graphic_context[graphiccontextindex]->render=symbolDepth > 0 ? MagickFalse :
                  MagickTrue;
                break;
              }
            status=MagickFalse;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'r':
      case 'R':
      {
        if (LocaleCompare("rectangle",keyword) == 0)
          {
            primitive_type=RectanglePrimitive;
            break;
          }
        if (LocaleCompare("rotate",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            angle=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            affine.sx=cos(DegreesToRadians(fmod((double) angle,360.0)));
            affine.rx=sin(DegreesToRadians(fmod((double) angle,360.0)));
            affine.ry=(-sin(DegreesToRadians(fmod((double) angle,360.0))));
            affine.sy=cos(DegreesToRadians(fmod((double) angle,360.0)));
            break;
          }
        if (LocaleCompare("roundRectangle",keyword) == 0)
          {
            primitive_type=RoundRectanglePrimitive;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 's':
      case 'S':
      {
        if (LocaleCompare("scale",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            affine.sx=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            affine.sy=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        if (LocaleCompare("skewX",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            angle=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            affine.ry=sin(DegreesToRadians(angle));
            break;
          }
        if (LocaleCompare("skewY",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            angle=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            affine.rx=(-tan(DegreesToRadians(angle)/2.0));
            break;
          }
        if (LocaleCompare("stop-color",keyword) == 0)
          {
            PixelInfo
              stop_color;

            number_stops++;
            if (number_stops == 1)
              stops=(StopInfo *) AcquireQuantumMemory(2,sizeof(*stops));
            else
              if (number_stops > 2)
                stops=(StopInfo *) ResizeQuantumMemory(stops,number_stops,
                  sizeof(*stops));
            if (stops == (StopInfo *) NULL)
              {
                (void) ThrowMagickException(exception,GetMagickModule(),
                  ResourceLimitError,"MemoryAllocationFailed","`%s'",
                  image->filename);
                break;
              }
            (void) GetNextToken(q,&q,extent,token);
            status&=QueryColorCompliance(token,AllCompliance,&stop_color,
              exception);
            stops[number_stops-1].color=stop_color;
            (void) GetNextToken(q,&q,extent,token);
            factor=strchr(token,'%') != (char *) NULL ? 0.01 : 1.0;
            stops[number_stops-1].offset=factor*GetDrawValue(token,
              &next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        if (LocaleCompare("stroke",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            if (graphic_context[graphiccontextindex]->clip_path != MagickFalse)
              break;
            (void) FormatLocaleString(pattern,MagickPathExtent,"%s",token);
            if (GetImageArtifact(image,pattern) != (const char *) NULL)
              (void) DrawPatternPath(image,draw_info,token,
                &graphic_context[graphiccontextindex]->stroke_pattern,exception);
            else
              {
                status&=QueryColorCompliance(token,AllCompliance,
                  &graphic_context[graphiccontextindex]->stroke,exception);
                if (graphic_context[graphiccontextindex]->stroke_alpha != OpaqueAlpha)
                  graphic_context[graphiccontextindex]->stroke.alpha=
                    graphic_context[graphiccontextindex]->stroke_alpha;
               }
            break;
          }
        if (LocaleCompare("stroke-antialias",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->stroke_antialias=StringToLong(token) != 0 ?
              MagickTrue : MagickFalse;
            break;
          }
        if (LocaleCompare("stroke-dasharray",keyword) == 0)
          {
            if (graphic_context[graphiccontextindex]->dash_pattern != (double *) NULL)
              graphic_context[graphiccontextindex]->dash_pattern=(double *)
                RelinquishMagickMemory(graphic_context[graphiccontextindex]->dash_pattern);
            if (IsPoint(q) != MagickFalse)
              {
                const char
                  *r;

                r=q;
                (void) GetNextToken(r,&r,extent,token);
                if (*token == ',')
                  (void) GetNextToken(r,&r,extent,token);
                for (x=0; IsPoint(token) != MagickFalse; x++)
                {
                  (void) GetNextToken(r,&r,extent,token);
                  if (*token == ',')
                    (void) GetNextToken(r,&r,extent,token);
                }
                graphic_context[graphiccontextindex]->dash_pattern=(double *)
                  AcquireQuantumMemory((size_t) (2*x+2),
                  sizeof(*graphic_context[graphiccontextindex]->dash_pattern));
                if (graphic_context[graphiccontextindex]->dash_pattern == (double *) NULL)
                  {
                    (void) ThrowMagickException(exception,GetMagickModule(),
                      ResourceLimitError,"MemoryAllocationFailed","`%s'",
                      image->filename);
                    status=MagickFalse;
                    break;
                  }
                (void) memset(graphic_context[graphiccontextindex]->dash_pattern,0,(size_t)
                  (2*x+2)*sizeof(*graphic_context[graphiccontextindex]->dash_pattern));
                for (j=0; j < x; j++)
                {
                  (void) GetNextToken(q,&q,extent,token);
                  if (*token == ',')
                    (void) GetNextToken(q,&q,extent,token);
                  graphic_context[graphiccontextindex]->dash_pattern[j]=GetDrawValue(token,
                    &next_token);
                  if (token == next_token)
                    ThrowPointExpectedException(token,exception);
                  if (graphic_context[graphiccontextindex]->dash_pattern[j] < 0.0)
                    status=MagickFalse;
                }
                if ((x & 0x01) != 0)
                  for ( ; j < (2*x); j++)
                    graphic_context[graphiccontextindex]->dash_pattern[j]=
                      graphic_context[graphiccontextindex]->dash_pattern[j-x];
                graphic_context[graphiccontextindex]->dash_pattern[j]=0.0;
                break;
              }
            (void) GetNextToken(q,&q,extent,token);
            break;
          }
        if (LocaleCompare("stroke-dashoffset",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->dash_offset=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        if (LocaleCompare("stroke-linecap",keyword) == 0)
          {
            ssize_t
              linecap;

            (void) GetNextToken(q,&q,extent,token);
            linecap=ParseCommandOption(MagickLineCapOptions,MagickFalse,token);
            if (linecap == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->linecap=(LineCap) linecap;
            break;
          }
        if (LocaleCompare("stroke-linejoin",keyword) == 0)
          {
            ssize_t
              linejoin;

            (void) GetNextToken(q,&q,extent,token);
            linejoin=ParseCommandOption(MagickLineJoinOptions,MagickFalse,
              token);
            if (linejoin == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->linejoin=(LineJoin) linejoin;
            break;
          }
        if (LocaleCompare("stroke-miterlimit",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->miterlimit=StringToUnsignedLong(token);
            break;
          }
        if (LocaleCompare("stroke-opacity",keyword) == 0)
          {
            double
              opacity;

            (void) GetNextToken(q,&q,extent,token);
            if (graphic_context[graphiccontextindex]->clip_path != MagickFalse)
              break;
            factor=strchr(token,'%') != (char *) NULL ? 0.01 : 1.0;
            opacity=MagickMin(MagickMax(factor*
              GetDrawValue(token,&next_token),0.0),1.0);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            if (graphic_context[graphiccontextindex]->compliance == SVGCompliance)
              graphic_context[graphiccontextindex]->stroke_alpha*=opacity;
            else
              graphic_context[graphiccontextindex]->stroke_alpha=QuantumRange*opacity;
            if (graphic_context[graphiccontextindex]->stroke.alpha != TransparentAlpha)
              graphic_context[graphiccontextindex]->stroke.alpha=graphic_context[graphiccontextindex]->stroke_alpha;
            else
              graphic_context[graphiccontextindex]->stroke.alpha=(MagickRealType)
                ClampToQuantum(QuantumRange*(1.0-opacity));
            break;
          }
        if (LocaleCompare("stroke-width",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            if (graphic_context[graphiccontextindex]->clip_path != MagickFalse)
              break;
            graphic_context[graphiccontextindex]->stroke_width=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        status=MagickFalse;
        break;
      }
      case 't':
      case 'T':
      {
        if (LocaleCompare("text",keyword) == 0)
          {
            primitive_type=TextPrimitive;
            cursor=0.0;
            break;
          }
        if (LocaleCompare("text-align",keyword) == 0)
          {
            ssize_t
              align;

            (void) GetNextToken(q,&q,extent,token);
            align=ParseCommandOption(MagickAlignOptions,MagickFalse,token);
            if (align == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->align=(AlignType) align;
            break;
          }
        if (LocaleCompare("text-anchor",keyword) == 0)
          {
            ssize_t
              align;

            (void) GetNextToken(q,&q,extent,token);
            align=ParseCommandOption(MagickAlignOptions,MagickFalse,token);
            if (align == -1)
              {
                status=MagickFalse;
                break;
              }
            graphic_context[graphiccontextindex]->align=(AlignType) align;
            break;
          }
        if (LocaleCompare("text-antialias",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->text_antialias=StringToLong(token) != 0 ?
              MagickTrue : MagickFalse;
            break;
          }
        if (LocaleCompare("text-undercolor",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            status&=QueryColorCompliance(token,AllCompliance,
              &graphic_context[graphiccontextindex]->undercolor,exception);
            break;
          }
        if (LocaleCompare("translate",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            affine.tx=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            affine.ty=GetDrawValue(token,&next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            cursor=0.0;
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'u':
      case 'U':
      {
        if (LocaleCompare("use",keyword) == 0)
          {
            const char
              *use;

            /*
              Get a macro from the MVG document, and "use" it here.
            */
            (void) GetNextToken(q,&q,extent,token);
            use=(const char *) GetValueFromSplayTree(macros,token);
            if (use != (const char *) NULL)
              {
                clone_info=CloneDrawInfo((ImageInfo *) NULL,graphic_context[graphiccontextindex]);
                (void) CloneString(&clone_info->primitive,use);
                status=RenderMVGContent(image,clone_info,depth+1,exception);
                clone_info=DestroyDrawInfo(clone_info);
              }
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'v':
      case 'V':
      {
        if (LocaleCompare("viewbox",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->viewbox.x=(ssize_t) ceil(GetDrawValue(token,
              &next_token)-0.5);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->viewbox.y=(ssize_t) ceil(GetDrawValue(token,
              &next_token)-0.5);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->viewbox.width=(size_t) floor(GetDrawValue(
              token,&next_token)+0.5);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            (void) GetNextToken(q,&q,extent,token);
            if (*token == ',')
              (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->viewbox.height=(size_t) floor(GetDrawValue(
              token,&next_token)+0.5);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        status=MagickFalse;
        break;
      }
      case 'w':
      case 'W':
      {
        if (LocaleCompare("word-spacing",keyword) == 0)
          {
            (void) GetNextToken(q,&q,extent,token);
            graphic_context[graphiccontextindex]->interword_spacing=GetDrawValue(token,
              &next_token);
            if (token == next_token)
              ThrowPointExpectedException(token,exception);
            break;
          }
        status=MagickFalse;
        break;
      }
      default:
      {
        status=MagickFalse;
        break;
      }
    }
    if (status == MagickFalse)
      break;
    if ((fabs(affine.sx-1.0) >= MagickEpsilon) ||
        (fabs(affine.rx) >= MagickEpsilon) || (fabs(affine.ry) >= MagickEpsilon) ||
        (fabs(affine.sy-1.0) >= MagickEpsilon) ||
        (fabs(affine.tx) >= MagickEpsilon) || (fabs(affine.ty) >= MagickEpsilon))
      {
        graphic_context[graphiccontextindex]->affine.sx=current.sx*affine.sx+current.ry*affine.rx;
        graphic_context[graphiccontextindex]->affine.rx=current.rx*affine.sx+current.sy*affine.rx;
        graphic_context[graphiccontextindex]->affine.ry=current.sx*affine.ry+current.ry*affine.sy;
        graphic_context[graphiccontextindex]->affine.sy=current.rx*affine.ry+current.sy*affine.sy;
        graphic_context[graphiccontextindex]->affine.tx=current.sx*affine.tx+current.ry*affine.ty+
          current.tx;
        graphic_context[graphiccontextindex]->affine.ty=current.rx*affine.tx+current.sy*affine.ty+
          current.ty;
      }
    if (primitive_type == UndefinedPrimitive)
      {
        if (*q == '\0')
          {
            if (number_stops > 1)
              {
                GradientType
                  type;

              type=LinearGradient;
              if (draw_info->gradient.type == RadialGradient)
                type=RadialGradient;
              (void) GradientImage(image,type,PadSpread,stops,number_stops,
                exception);
             }
           if (number_stops > 0)
             stops=(StopInfo *) RelinquishMagickMemory(stops);
          }
        if ((image->debug != MagickFalse) && (q > p))
          (void) LogMagickEvent(DrawEvent,GetMagickModule(),"  %.*s",(int)
            (q-p-1),p);
        continue;
      }
    /*
      Parse the primitive attributes.
    */
    for (i=0; primitive_info[i].primitive != UndefinedPrimitive; i++)
      if ((primitive_info[i].primitive == TextPrimitive) ||
          (primitive_info[i].primitive == ImagePrimitive))
        if (primitive_info[i].text != (char *) NULL)
          primitive_info[i].text=DestroyString(primitive_info[i].text);
    i=0;
    mvg_info.offset=i;
    j=0;
    primitive_info[0].point.x=0.0;
    primitive_info[0].point.y=0.0;
    primitive_info[0].coordinates=0;
    primitive_info[0].method=FloodfillMethod;
    primitive_info[0].closed_subpath=MagickFalse;
    for (x=0; *q != '\0'; x++)
    {
      /*
        Define points.
      */
      if (IsPoint(q) == MagickFalse)
        break;
      (void) GetNextToken(q,&q,extent,token);
      point.x=GetDrawValue(token,&next_token);
      if (token == next_token)
        ThrowPointExpectedException(token,exception);
      (void) GetNextToken(q,&q,extent,token);
      if (*token == ',')
        (void) GetNextToken(q,&q,extent,token);
      point.y=GetDrawValue(token,&next_token);
      if (token == next_token)
        ThrowPointExpectedException(token,exception);
      (void) GetNextToken(q,(const char **) NULL,extent,token);
      if (*token == ',')
        (void) GetNextToken(q,&q,extent,token);
      primitive_info[i].primitive=primitive_type;
      primitive_info[i].point=point;
      primitive_info[i].coordinates=0;
      primitive_info[i].method=FloodfillMethod;
      primitive_info[i].closed_subpath=MagickFalse;
      i++;
      mvg_info.offset=i;
      if (i < (ssize_t) number_points)
        continue;
      status&=CheckPrimitiveExtent(&mvg_info,number_points);
    }
    if (status == MagickFalse)
      break;
    if ((primitive_info[j].primitive == TextPrimitive) ||
        (primitive_info[j].primitive == ImagePrimitive))
      if (primitive_info[j].text != (char *) NULL)
        primitive_info[j].text=DestroyString(primitive_info[j].text);
    primitive_info[j].primitive=primitive_type;
    primitive_info[j].coordinates=(size_t) x;
    primitive_info[j].method=FloodfillMethod;
    primitive_info[j].closed_subpath=MagickFalse;
    /*
      Circumscribe primitive within a circle.
    */
    bounds.x1=primitive_info[j].point.x;
    bounds.y1=primitive_info[j].point.y;
    bounds.x2=primitive_info[j].point.x;
    bounds.y2=primitive_info[j].point.y;
    for (k=1; k < (ssize_t) primitive_info[j].coordinates; k++)
    {
      point=primitive_info[j+k].point;
      if (point.x < bounds.x1)
        bounds.x1=point.x;
      if (point.y < bounds.y1)
        bounds.y1=point.y;
      if (point.x > bounds.x2)
        bounds.x2=point.x;
      if (point.y > bounds.y2)
        bounds.y2=point.y;
    }
    /*
      Speculate how many points our primitive might consume.
    */
    coordinates=(double) primitive_info[j].coordinates;
    switch (primitive_type)
    {
      case RectanglePrimitive:
      {
        coordinates*=5.0;
        break;
      }
      case RoundRectanglePrimitive:
      {
        double
          alpha,
          beta,
          radius;

        alpha=bounds.x2-bounds.x1;
        beta=bounds.y2-bounds.y1;
        radius=hypot((double) alpha,(double) beta);
        coordinates*=5.0;
        coordinates+=2.0*((size_t) ceil((double) MagickPI*radius))+6.0*
          BezierQuantum+360.0;
        break;
      }
      case BezierPrimitive:
      {
        coordinates=(double) (BezierQuantum*primitive_info[j].coordinates);
        if (primitive_info[j].coordinates > (107*BezierQuantum))
          {
            (void) ThrowMagickException(exception,GetMagickModule(),DrawError,
              "TooManyBezierCoordinates","`%s'",token);
            status=MagickFalse;
            break;
          }
        break;
      }
      case PathPrimitive:
      {
        char
          *s,
          *t;

        (void) GetNextToken(q,&q,extent,token);
        coordinates=1.0;
        t=token;
        for (s=token; *s != '\0'; s=t)
        {
          double
            value;

          value=GetDrawValue(s,&t);
          (void) value;
          if (s == t)
            {
              t++;
              continue;
            }
          coordinates++;
        }
        for (s=token; *s != '\0'; s++)
          if (strspn(s,"AaCcQqSsTt") != 0)
            coordinates+=(20.0*BezierQuantum)+360.0;
        break;
      }
      case CirclePrimitive:
      case ArcPrimitive:
      case EllipsePrimitive:
      {
        double
          alpha,
          beta,
          radius;

        alpha=bounds.x2-bounds.x1;
        beta=bounds.y2-bounds.y1;
        radius=hypot(alpha,beta);
        coordinates=2.0*(ceil(MagickPI*radius))+6.0*BezierQuantum+360.0;
        if (coordinates > (107*BezierQuantum))
          {
            (void) ThrowMagickException(exception,GetMagickModule(),DrawError,
              "TooManyBezierCoordinates","`%s'",token);
            status=MagickFalse;
            break;
          }
        break;
      }
      default:
        break;
    }
    if (status == MagickFalse)
      break;
    if (((size_t) (i+coordinates)) >= number_points)
      {
        /*
          Resize based on speculative points required by primitive.
        */
        number_points+=coordinates+1;
        if (number_points < (size_t) coordinates)
          {
            (void) ThrowMagickException(exception,GetMagickModule(),
              ResourceLimitError,"MemoryAllocationFailed","`%s'",
              image->filename);
            break;
          }
        mvg_info.offset=i;
        status&=CheckPrimitiveExtent(&mvg_info,number_points);
      }
    status&=CheckPrimitiveExtent(&mvg_info,PrimitiveExtentPad);
    if (status == MagickFalse)
      break;
    mvg_info.offset=j;
    switch (primitive_type)
    {
      case PointPrimitive:
      default:
      {
        if (primitive_info[j].coordinates != 1)
          {
            status=MagickFalse;
            break;
          }
        status&=TracePoint(primitive_info+j,primitive_info[j].point);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case LinePrimitive:
      {
        double
          dx,
          dy,
          maximum_length;

        if (primitive_info[j].coordinates != 2)
          {
            status=MagickFalse;
            break;
          }
        dx=primitive_info[i].point.x-primitive_info[i-1].point.x;
        dy=primitive_info[i].point.y-primitive_info[i-1].point.y;
        maximum_length=hypot(dx,dy);
        if (maximum_length > (MaxBezierCoordinates/100.0))
          ThrowPointExpectedException(keyword,exception);
        status&=TraceLine(primitive_info+j,primitive_info[j].point,
          primitive_info[j+1].point);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case RectanglePrimitive:
      {
        if (primitive_info[j].coordinates != 2)
          {
            status=MagickFalse;
            break;
          }
        status&=TraceRectangle(primitive_info+j,primitive_info[j].point,
          primitive_info[j+1].point);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case RoundRectanglePrimitive:
      {
        if (primitive_info[j].coordinates != 3)
          {
            status=MagickFalse;
            break;
          }
        if ((primitive_info[j+2].point.x < 0.0) ||
            (primitive_info[j+2].point.y < 0.0))
          {
            status=MagickFalse;
            break;
          }
        if ((primitive_info[j+1].point.x-primitive_info[j].point.x) < 0.0)
          {
            status=MagickFalse;
            break;
          }
        if ((primitive_info[j+1].point.y-primitive_info[j].point.y) < 0.0)
          {
            status=MagickFalse;
            break;
          }
        status&=TraceRoundRectangle(&mvg_info,primitive_info[j].point,
          primitive_info[j+1].point,primitive_info[j+2].point);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case ArcPrimitive:
      {
        if (primitive_info[j].coordinates != 3)
          {
            status=MagickFalse;
            break;
          }
        status&=TraceArc(&mvg_info,primitive_info[j].point,
          primitive_info[j+1].point,primitive_info[j+2].point);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case EllipsePrimitive:
      {
        if (primitive_info[j].coordinates != 3)
          {
            status=MagickFalse;
            break;
          }
        if ((primitive_info[j+1].point.x < 0.0) ||
            (primitive_info[j+1].point.y < 0.0))
          {
            status=MagickFalse;
            break;
          }
        status&=TraceEllipse(&mvg_info,primitive_info[j].point,
          primitive_info[j+1].point,primitive_info[j+2].point);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case CirclePrimitive:
      {
        if (primitive_info[j].coordinates != 2)
          {
            status=MagickFalse;
            break;
          }
        status&=TraceCircle(&mvg_info,primitive_info[j].point,
          primitive_info[j+1].point);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case PolylinePrimitive:
      {
        if (primitive_info[j].coordinates < 1)
          {
            status=MagickFalse;
            break;
          }
        break;
      }
      case PolygonPrimitive:
      {
        if (primitive_info[j].coordinates < 3)
          {
            status=MagickFalse;
            break;
          }
        primitive_info[i]=primitive_info[j];
        primitive_info[i].coordinates=0;
        primitive_info[j].coordinates++;
        primitive_info[j].closed_subpath=MagickTrue;
        i++;
        break;
      }
      case BezierPrimitive:
      {
        if (primitive_info[j].coordinates < 3)
          {
            status=MagickFalse;
            break;
          }
        status&=TraceBezier(&mvg_info,primitive_info[j].coordinates);
        i=(ssize_t) (j+primitive_info[j].coordinates);
        break;
      }
      case PathPrimitive:
      {
        coordinates=(double) TracePath(&mvg_info,token,exception);
        if (coordinates < 0.0)
          {
            status=MagickFalse;
            break;
          }
        i=(ssize_t) (j+coordinates);
        break;
      }
      case AlphaPrimitive:
      case ColorPrimitive:
      {
        ssize_t
          method;

        if (primitive_info[j].coordinates != 1)
          {
            status=MagickFalse;
            break;
          }
        (void) GetNextToken(q,&q,extent,token);
        method=ParseCommandOption(MagickMethodOptions,MagickFalse,token);
        if (method == -1)
          {
            status=MagickFalse;
            break;
          }
        primitive_info[j].method=(PaintMethod) method;
        break;
      }
      case TextPrimitive:
      {
        char
          geometry[MagickPathExtent];

        if (primitive_info[j].coordinates != 1)
          {
            status=MagickFalse;
            break;
          }
        if (*token != ',')
          (void) GetNextToken(q,&q,extent,token);
        (void) CloneString(&primitive_info[j].text,token);
        /*
          Compute text cursor offset.
        */
        clone_info=CloneDrawInfo((ImageInfo *) NULL,graphic_context[graphiccontextindex]);
        if ((fabs(mvg_info.point.x-primitive_info->point.x) < MagickEpsilon) &&
            (fabs(mvg_info.point.y-primitive_info->point.y) < MagickEpsilon))
          {
            mvg_info.point=primitive_info->point;
            primitive_info->point.x+=cursor;
          }
        else
          {
            mvg_info.point=primitive_info->point;
            cursor=0.0;
          }
        (void) FormatLocaleString(geometry,MagickPathExtent,"%+f%+f",
          primitive_info->point.x,primitive_info->point.y);
        clone_info->render=MagickFalse;
        clone_info->text=AcquireString(token);
        status&=GetTypeMetrics(image,clone_info,&metrics,exception);
        clone_info=DestroyDrawInfo(clone_info);
        cursor+=metrics.width;
        if (graphic_context[graphiccontextindex]->compliance != SVGCompliance)
          cursor=0.0;
        break;
      }
      case ImagePrimitive:
      {
        if (primitive_info[j].coordinates != 2)
          {
            status=MagickFalse;
            break;
          }
        (void) GetNextToken(q,&q,extent,token);
        (void) CloneString(&primitive_info[j].text,token);
        break;
      }
    }
    mvg_info.offset=i;
    if (status == 0)
      break;
    primitive_info[i].primitive=UndefinedPrimitive;
    if ((image->debug != MagickFalse) && (q > p))
      (void) LogMagickEvent(DrawEvent,GetMagickModule(),"  %.*s",(int) (q-p-1),
        p);
    /*
      Sanity check.
    */
    status&=CheckPrimitiveExtent(&mvg_info,
      ExpandAffine(&graphic_context[graphiccontextindex]->affine));
    if (status == 0)
      break;
    status&=CheckPrimitiveExtent(&mvg_info,graphic_context[graphiccontextindex]->stroke_width);
    if (status == 0)
      break;
    if (i == 0)
      continue;
    /*
      Transform points.
    */
    for (i=0; primitive_info[i].primitive != UndefinedPrimitive; i++)
    {
      point=primitive_info[i].point;
      primitive_info[i].point.x=graphic_context[graphiccontextindex]->affine.sx*point.x+
        graphic_context[graphiccontextindex]->affine.ry*point.y+graphic_context[graphiccontextindex]->affine.tx;
      primitive_info[i].point.y=graphic_context[graphiccontextindex]->affine.rx*point.x+
        graphic_context[graphiccontextindex]->affine.sy*point.y+graphic_context[graphiccontextindex]->affine.ty;
      point=primitive_info[i].point;
      if (point.x < graphic_context[graphiccontextindex]->bounds.x1)
        graphic_context[graphiccontextindex]->bounds.x1=point.x;
      if (point.y < graphic_context[graphiccontextindex]->bounds.y1)
        graphic_context[graphiccontextindex]->bounds.y1=point.y;
      if (point.x > graphic_context[graphiccontextindex]->bounds.x2)
        graphic_context[graphiccontextindex]->bounds.x2=point.x;
      if (point.y > graphic_context[graphiccontextindex]->bounds.y2)
        graphic_context[graphiccontextindex]->bounds.y2=point.y;
      if (primitive_info[i].primitive == ImagePrimitive)
        break;
      if (i >= (ssize_t) number_points)
        ThrowFatalException(ResourceLimitFatalError,"MemoryAllocationFailed");
    }
    if (graphic_context[graphiccontextindex]->render != MagickFalse)
      {
        if ((graphiccontextindex != 0) && (graphic_context[graphiccontextindex]->compliance != SVGCompliance) &&
            (graphic_context[graphiccontextindex]->clip_mask != (char *) NULL) &&
            (LocaleCompare(graphic_context[graphiccontextindex]->clip_mask,
             graphic_context[graphiccontextindex-1]->clip_mask) != 0))
          {
            const char
              *clip_path;

            clip_path=(const char *) GetValueFromSplayTree(macros,
              graphic_context[graphiccontextindex]->clip_mask);
            if (clip_path != (const char *) NULL)
              (void) SetImageArtifact(image,graphic_context[graphiccontextindex]->clip_mask,
                clip_path);
            status&=DrawClipPath(image,graphic_context[graphiccontextindex],
              graphic_context[graphiccontextindex]->clip_mask,exception);
          }
        status&=DrawPrimitive(image,graphic_context[graphiccontextindex],primitive_info,
          exception);
      }
    proceed=SetImageProgress(image,RenderImageTag,q-primitive,(MagickSizeType)
      primitive_extent);
    if (proceed == MagickFalse)
      break;
    if (status == 0)
      break;
  }
  if (image->debug != MagickFalse)
    (void) LogMagickEvent(DrawEvent,GetMagickModule(),"end draw-image");
  /*
    Relinquish resources.
  */
  macros=DestroySplayTree(macros);
  token=DestroyString(token);
  if (primitive_info != (PrimitiveInfo *) NULL)
    {
      for (i=0; primitive_info[i].primitive != UndefinedPrimitive; i++)
        if ((primitive_info[i].primitive == TextPrimitive) ||
            (primitive_info[i].primitive == ImagePrimitive))
          if (primitive_info[i].text != (char *) NULL)
            primitive_info[i].text=DestroyString(primitive_info[i].text);
      primitive_info=(PrimitiveInfo *) RelinquishMagickMemory(primitive_info);
    }
  primitive=DestroyString(primitive);
  if (stops != (StopInfo *) NULL)
    stops=(StopInfo *) RelinquishMagickMemory(stops);
  for ( ; graphiccontextindex >= 0; graphiccontextindex--)
    graphic_context[graphiccontextindex]=DestroyDrawInfo(graphic_context[graphiccontextindex]);
  graphic_context=(DrawInfo **) RelinquishMagickMemory(graphic_context);
  if (status == MagickFalse)
    ThrowBinaryException(DrawError,"NonconformingDrawingPrimitiveDefinition",
      keyword);
  return(status != 0 ? MagickTrue : MagickFalse);
}
