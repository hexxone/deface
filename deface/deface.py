#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
from typing import Dict, Tuple

import tqdm
import skimage.draw
import numpy as np
import imageio.v3 as iio
import cv2
import re

from deface import __version__
from deface.centerface import CenterFace
from deface.track import Tracker


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg = None,
        mosaicsize: int = 20,
        feathering: float = 0.1,
        alpha: float = 1.0
):
    h, w = y2 - y1, x2 - x1
    if (w <= 0 or h <= 0): return

    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur' and alpha > 0:
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)

        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (max(1, int(alpha * w // bf)), max(1, int(alpha * h // bf)))
        )
        if ellipse and h >= 2 and w >= 2:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse(h // 2, w // 2, h // 2, w // 2)

            # Create a feathered mask
            mask = np.zeros_like(roibox, dtype=float)
            mask[ey, ex] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * feathering, sigmaY=h * feathering)

            roibox = roibox * (1 - mask) + blurred_box * mask
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box

    elif replacewith == 'img':
        target_size = (w, h)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)

    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)

    elif replacewith == 'none':
        pass

    if draw_scores:
        cv2.putText(
            frame, f'{score:.2f}', (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
        )


def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize, feathering
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        alpha = det[5] if len(det) > 5 else 1
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1 = max(0, min(frame.shape[0] - 1, y1))
        y2 = max(0, min(frame.shape[0] - 1, y2))
        x1 = max(0, min(frame.shape[1] - 1, x1))
        x2 = max(0, min(frame.shape[1] - 1, x2))

        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize,
            feathering=feathering,
            alpha=alpha
        )


def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()


def video_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        ffmpeg_config: Dict[str, str],
        replaceimg = None,
        keep_audio: bool = False,
        mosaicsize: int = 20,
        disable_progress_output = False,
        unstable: bool = False,
        smoothing_window: int = 5,
        feathering: float = 0.1,
        persist_last_pos: bool = False,
        persist_with_tracking: bool = False,
        fade_in: int = 5,
        fade_out: int = 5,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.5
):
    try:
        meta = iio.immeta(ipath, plugin='pyav')
    except Exception as error:
        if cam:
            print(f'Could not find video device {ipath}: {error}. Please set a valid input.')
        else:
            print(f'Could not open file {ipath} as a video file with imageio: {error}. Skipping file...')
        return

    if cam:
        nframes = None
    else:
        nframes = iio.improps(ipath, plugin='pyav').shape[0]
    if nested:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, position=1, leave=True, disable=disable_progress_output)
    else:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, disable=disable_progress_output)

    if opath is not None:
        _ffmpeg_config = ffmpeg_config.copy()
        # If fps is not explicitly set in ffmpeg_config, use source video fps value
        # https://github.com/imageio/imageio/issues/1120
        approximate_fps = round(meta['fps'], 1)
        _ffmpeg_config.setdefault('fps', approximate_fps)
        # Carry over audio from input path, use "copy" codec (no transcoding) by default
        if keep_audio and meta.get('audio_codec'):
            _ffmpeg_config.setdefault('audio_path', ipath)
            _ffmpeg_config.setdefault('audio_codec', 'copy')
        writer = iio.imopen(opath, 'w', plugin='pyav')
        writer.init_video_stream(**_ffmpeg_config)
        # Workaround for https://github.com/imageio/imageio/issues/1139
        writer._container.streams.video[0].codec_context.time_base = writer._container.streams.video[0].time_base

    tracker = Tracker(
        max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold,
        smoothing_window=smoothing_window,
        fade_in=fade_in, fade_out=fade_out
    )
    for frame in iio.imiter(ipath, plugin='pyav'):
        # Perform network inference, get bb dets but discard landmark predictions
        dets, _ = centerface(frame, threshold=threshold)

        if unstable:
            anonymize_frame(
                dets, frame, mask_scale=mask_scale,
                replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
                replaceimg=replaceimg, mosaicsize=mosaicsize, feathering=feathering
            )
        else:
            # Update tracker
            trackers = tracker.update(dets)

            bboxes = []
            for trk in trackers:
                if (trk.time_since_update <= 1):
                    bboxes.append(trk.smoothed_hit_bbox)
                else:
                    if (persist_last_pos):
                        bboxes.append(trk.smoothed_hit_bbox)
                    if (persist_with_tracking):
                        bboxes.append(trk.predicted_bbox)

            # Anonymize using tracked bbs
            anonymize_frame(
                bboxes, frame, mask_scale=mask_scale,
                replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
                replaceimg=replaceimg, mosaicsize=mosaicsize, feathering=feathering
            )


        if opath is not None:
            writer.write_frame(frame)

        if enable_preview:
            cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
                cv2.destroyAllWindows()
                break
        bar.update()

    if opath is not None:
        writer.close()
    bar.close()


def image_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg = None,
        mosaicsize: int = 20,
        feathering: float = 0.1
):
    frame = iio.imread(ipath)

    if keep_metadata:
        # Source image EXIF metadata retrieval via imageio V3 lib
        metadata = iio.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    # Perform network inference, get bb dets but discard landmark predictions
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize, feathering=feathering
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
            cv2.destroyAllWindows()


    if keep_metadata:
        # Save image with EXIF metadata
        iio.imwrite(opath, frame, exif=exif_dict)
    else:
        iio.imwrite(opath, frame)

    # print(f'Output saved to {opath}')


#  https://gist.github.com/bthaman/64b20ef47b2364b16c2c6bc529b1d451
def get_download_path():
    """Returns the default downloads path for linux, windows and macos"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'downloads')


def get_path_infos(path):
    url_pattern = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}(?:\\.[a-zA-Z0-9()]{1,6})?\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
    path_type = None
    media_type = None
    path_base = None
    path_filename = None
    path_ext = None

    if path.startswith('<video'):
        path_type = 'cam'
        media_type = 'video'
    elif re.match(url_pattern, path):
        path_type = 'url'
        path = re.split('[?#]', path)[0]
    elif os.path.isfile(path):
        path_type = 'file'
    elif os.path.isdir(path):
        path_type = 'dir'

    if path_type == 'dir':
        path_base = path
    else:
        path_base = os.path.dirname(path)
        path_filename, path_ext = os.path.splitext(os.path.basename(path))

    if path_type == None and not path_ext:
        path_base = path
        path_filename = None

    mime = mimetypes.guess_type(path)[0]
    if mime is not None:
        if mime.startswith('video'):
            media_type = 'video'
        elif mime.startswith('image'):
            media_type = 'image'

    return (path_type, media_type, path_base, path_filename, path_ext, mime)


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg = None
                         ):
    """
    Method for getting an anonymized image without CLI
    returns frame
    """

    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg
    )

    return frame


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help=f'File path(s), url(s) or camera device name. It is possible to pass multiple paths by separating them by spaces or by using shell expansion (e.g. `$ deface vids/*.mp4`). Alternatively, you can pass a directory as an input, in which case all files in the directory will be used as inputs. If a camera is installed, a live webcam demo can be started by running `$ deface cam` (which is a shortcut for `$ deface -p \'<video0>\'`.')
    parser.add_argument(
        '--output', '-o', default=None, metavar='O',
        help='Output file name. Defaults to input path with postfix "_anonymized". If input is a camera or url, defaults to the system downloads folder.')
    parser.add_argument(
        '--thresh', '-t', default=0.2, type=float, metavar='T',
        help='Detection threshold (tune this to trade off between false positive and false negative rate). Default: 0.2.')
    parser.add_argument(
        '--scale', '-s', default=None, metavar='WxH',
        help='Downscale images for network inference to this size (format: WxH, example: --scale 640x360).')
    parser.add_argument(
        '--preview', '-p', default=False, action='store_true',
        help='Enable live preview GUI (can decrease performance).')
    parser.add_argument(
        '--boxes', default=False, action='store_true',
        help='Use boxes instead of ellipse masks.')
    parser.add_argument(
        '--draw-scores', default=False, action='store_true',
        help='Draw detection scores onto outputs.')
    parser.add_argument(
        '--disable-progress-output', default=False, action='store_true',
        help='Disable video progress output to console.')
    parser.add_argument(
        '--mask-scale', default=1.3, type=float, metavar='M',
        help='Scale factor for face masks, to make sure that masks cover the complete face. Default: 1.3.')
    parser.add_argument(
        '--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
        help='Anonymization filter mode for face regions. "blur" applies a strong gaussian blurring, "solid" draws a solid black box, "none" does leaves the input unchanged, "img" replaces the face with a custom image and "mosaic" replaces the face with mosaic. Default: "blur".')
    parser.add_argument(
        '--replaceimg', default='replace_img.png',
        help='Anonymization image for face regions. Requires --replacewith img option.')
    parser.add_argument(
        '--mosaicsize', default=20, type=int, metavar='width',
        help='Setting the mosaic size. Requires --replacewith mosaic option. Default: 20.')
    parser.add_argument(
        '--keep-audio', '-k', default=False, action='store_true',
        help='Keep audio from video source file and copy it over to the output (only applies to videos).')
    parser.add_argument(
        '--ffmpeg-config', default={"codec": "libx264"}, type=json.loads,
        help='FFMPEG config arguments for encoding output videos. This argument is expected in JSON notation. For a list of possible options, refer to the ffmpeg-imageio docs. Default: \'{"codec": "libx264"}\'.'
    )  # See https://imageio.readthedocs.io/en/stable/format_ffmpeg.html#parameters-for-saving
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
        help='Backend for ONNX model execution. Default: "auto" (prefer onnxrt if available).')
    parser.add_argument(
        '--execution-provider', '--ep', default=None, metavar='EP',
        help='Override onnxrt execution provider (see https://onnxruntime.ai/docs/execution-providers/). If not specified, the presumably fastest available one will be automatically selected. Only used if backend is onnxrt.')
    parser.add_argument(
        '--version', action='version', version=__version__,
        help='Print version number and exit.')
    parser.add_argument(
        '--unstable', default=False, action='store_true',
        help='Disable tracking and use unstable frame-by-frame detection.')
    parser.add_argument(
        '--feathering', default=0.1, type=float,
        help='Feathering amount for smooth mask borders. Default: 0.1.')
    parser.add_argument(
        '--persist-last-pos', default=False, action='store_true',
        help='Keep a mask when face is no longer detected, on the last known position.')
    parser.add_argument(
        '--persist-with-tracking', default=False, action='store_true',
        help='Keep a mask when face is no longer detected, predicting its position with bounding box tracking.')
    parser.add_argument(
        '--smoothing-window', default=5, type=int,
        help='Size of the smoothing window for bounding box tracking. Default: 5.')
    parser.add_argument(
        '--fade-in', default=5, type=int,
        help='Number of frames to fade in the mask. Default: 5.')
    parser.add_argument(
        '--fade-out', default=5, type=int,
        help='Number of frames to fade out the mask. Default: 5.')
    parser.add_argument(
        '--max-age', default=30, type=int,
        help='Maximum number of frames to keep a track without a detection. Default: 30.')
    parser.add_argument(
        '--min-hits', default=2, type=int,
        help='Minimum number of hits to start a track. Default: 2.')
    parser.add_argument(
        '--iou-threshold', default=0.5, type=float,
        help='IOU threshold for matching detections to tracks. Default: 0.5.')
    parser.add_argument(
        '--keep-metadata', '-m', default=True, action='store_true',
        help='Keep metadata of the original image. Default : True.')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')

    args = parser.parse_args()

    if len(args.input) == 0:
        parser.print_help()
        print('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:  # Shortcut for webcam demo with live preview
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    # add files in folders
    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path,file))
        else:
            # Either a path to a regular file, the special 'cam' shortcut
            # or an invalid path. The latter two cases are handled below.
            ipaths.append(path)


    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    keep_audio = args.keep_audio
    ffmpeg_config = args.ffmpeg_config
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata
    replaceimg = None
    disable_progress_output = args.disable_progress_output
    unstable = args.unstable
    smoothing_window = args.smoothing_window
    feathering = args.feathering
    persist_last_pos = args.persist_last_pos
    persist_with_tracking = args.persist_with_tracking
    fade_in = args.fade_in
    fade_out = args.fade_out
    max_age = args.max_age
    min_hits = args.min_hits
    iou_threshold = args.iou_threshold

    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == "img":
        replaceimg = iio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')


    # TODO: scalar downscaling setting (-> in_shape), preserving aspect ratio
    centerface = CenterFace(in_shape=in_shape, backend=backend, override_execution_provider=execution_provider)

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True

        path_type, media_type, path_base, path_filename, path_ext, _ = get_path_infos(ipath)

        if base_opath is None:
            if path_type != 'file':
                path_base = get_download_path()
            opath = os.path.join(path_base, f'{path_filename}_anonymized{path_ext}')
        else:
            opath_type, _, opath_base, opath_filename, _, _ = get_path_infos(base_opath)
            if opath_type is None and opath_base.length:
                os.makedirs(opath_base, exist_ok=True)
            if opath_filename is None:
                opath = os.path.join(opath_base, f'{path_filename}_anonymized{path_ext}')

        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not enable_preview:
            print('No output file is specified and the preview GUI is disabled. No output will be produced.')
        if media_type == 'video' or path_type == 'cam':
            video_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                cam=(path_type == 'cam'),
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                nested=multi_file,
                keep_audio=keep_audio,
                ffmpeg_config=ffmpeg_config,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                disable_progress_output=disable_progress_output,
                unstable=unstable,
                smoothing_window=smoothing_window,
                feathering=feathering,
                persist_last_pos=persist_last_pos,
                persist_with_tracking=persist_with_tracking,
                fade_in=fade_in,
                fade_out=fade_out,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold
            )
        elif media_type == 'image':
            image_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                keep_metadata=keep_metadata,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                feathering=feathering
            )
        elif media_type is None:
            print(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif media_type == 'notfound':
            print(f'File {ipath} not found. Skipping...')
        else:
            print(f'File {ipath} has an unknown type {media_type}. Skipping...')


if __name__ == '__main__':
    main()
