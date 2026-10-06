"""Batch text crops across frames using the existing Paddle OCR components."""


def predict_frames(ocr,images,check_cancel):
    if len(images)==1:
        return ocr.predict(images)
    pipeline=ocr.paddlex_pipeline
    if pipeline.use_doc_preprocessor or pipeline.use_textline_orientation or pipeline.text_type!='general':
        raise RuntimeError('当前批处理仅支持已配置的普通文字识别流程。')
    from paddlex.inference.pipelines.components import convert_points_to_boxes
    check_cancel()
    detected=list(pipeline.text_det_model(images))
    if len(detected)!=len(images):raise RuntimeError('画面文字检测结果数量不一致。')
    polys_by_frame=[];records=[]
    for frame,(image,result) in enumerate(zip(images,detected)):
        polys=pipeline._sort_boxes(result['dt_polys'])
        kept=[]
        for crop,poly in zip(pipeline._crop_by_polys(image,polys),polys):
            if crop.size>0 and crop.shape[0]>0 and crop.shape[1]>0:
                index=len(kept);kept.append(poly)
                records.append((frame,index,crop))
        polys_by_frame.append(kept)
    records.sort(key=lambda item:item[2].shape[1]/float(item[2].shape[0]))
    recognition={}
    if records:
        check_cancel()
        # Same recognition model and its configured six-crop batch limit.
        results=pipeline.text_rec_model([item[2] for item in records],return_word_box=False)
        count=0
        for item,result in zip(records,results):
            check_cancel()
            recognition[item[0],item[1]]=result
            count+=1
        if count!=len(records):raise RuntimeError('画面文字识别结果数量不一致。')
    output=[]
    for frame,polys in enumerate(polys_by_frame):
        texts=[];scores=[];accepted=[]
        for index,poly in enumerate(polys):
            result=recognition[frame,index]
            if result['rec_score']>=pipeline.text_rec_score_thresh:
                texts.append(result['rec_text']);scores.append(result['rec_score']);accepted.append(poly)
        output.append({'rec_texts':texts,'rec_scores':scores,'rec_boxes':convert_points_to_boxes(accepted)})
    return output
