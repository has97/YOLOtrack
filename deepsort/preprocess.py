import numpy as np
import cv2


def non_max_suppression(bbs, max_bb_ov, scores=None):

    if len(bbs) == 0: # number of boxes is zero
        return []

    bbs = bbs.astype(np.float)

    x1 = bbs[:, 0] # x coordinate lower
    y1 = bbs[:, 1] # y coordinate lower
    x2 = bbs[:, 2] + bbs[:, 0] # x coordinate higher
    y2 = bbs[:, 3] + bbs[:, 1] # y coordinate higher

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:# sorting by scores if not none
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)# else sort by higher y coordinate

    result = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last] # maximum score box
        result.append(i)

        # IOU calculation
        # maximum value for the bottom x and y coordinate for bottom (x,y) for overlap
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        # minimum value for the upper x and y coordinate for upper (x,y) for overlap
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # height and width of overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
         
        overlap = (w * h) / area[idxs[:last]]
        # delete all boxes with overlap greater than threshold that is defined and also the last box since it is already taken into account
        idxs = np.delete(idxs, np.concatenate( ([last], np.where(overlap > max_bb_ov)[0]) ) )

    return result # non max supressed boxes