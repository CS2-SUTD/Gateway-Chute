def bb_info(bbox, i):  ###
    width = float(bbox[i][2] - bbox[i][0])
    height = float(bbox[i][3] - bbox[i][1])
    return (
        float((bbox[i][2] + bbox[i][0]) / 2),
        float((bbox[i][3] + bbox[i][1]) / 2),
        height,
        width,
    )
