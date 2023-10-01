import cv2
import numpy as np

def interpolate_frames(frame1, frame2, target_frame_num):
    # Convert the input frames to grayscale for edge detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Find the edges in each frame using the Canny edge detector
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)

    # Calculate the number of frames to interpolate between the two input frames
    num_frames = abs(target_frame_num - 1)

    # Initialize an array to store the interpolated frames
    interpolated_frames = np.zeros((num_frames, frame1.shape[0], frame1.shape[1], frame1.shape[2]), dtype=np.uint8)

    # Interpolate the frames by linearly blending the edges between the two input frames
    for i in range(num_frames):
        # Calculate the blending coefficient
        alpha = (i + 1) / (num_frames + 1)

        # Linearly blend the edges using the calculated coefficient
        blended_edges = cv2.addWeighted(edges1, 1 - alpha, edges2, alpha, 0)

        # Convert the blended edges to a grayscale image
        blended_gray = cv2.cvtColor(blended_edges, cv2.COLOR_GRAY2BGR)

        # Use the blended edges to interpolate the color image
        interpolated_frames[i] = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

        # Apply the blended edges as a mask to the interpolated image
        interpolated_frames[i] = cv2.addWeighted(interpolated_frames[i], 1, blended_gray, alpha, 0)

    return interpolated_frames
def morph_frames(frame1, frame2, target_frame_num):
    # Define a set of corresponding points for the two input frames
    points1 = np.array([[0, 0], [0, frame1.shape[0] - 1], [frame1.shape[1] - 1, 0], [frame1.shape[1] - 1, frame1.shape[0] - 1]], dtype=np.float32)
    points2 = np.array([[0, 0], [0, frame2.shape[0] - 1], [frame2.shape[1] - 1, 0], [frame2.shape[1] - 1, frame2.shape[0] - 1]], dtype=np.float32)

    # Calculate the number of frames to interpolate between the two input frames
    num_frames = abs(target_frame_num - 1)

    # Initialize an array to store the interpolated frames
    interpolated_frames = np.zeros((num_frames, frame1.shape[0], frame1.shape[1], frame1.shape[2]), dtype=np.uint8)

    # Interpolate the frames by morphing the shapes of the input frames
    for i in range(num_frames):
        # Calculate the interpolation coefficient
        alpha = (i + 1) / (num_frames + 1)

        # Interpolate the corresponding points
        morphed_points = (1 - alpha) * points1 + alpha * points2

        # Calculate the affine transformation matrix from the corresponding points
        M = cv2.getAffineTransform(points1, morphed_points)

        # Apply the affine transformation to the first frame
        morphed_frame1 = cv2.warpAffine(frame1, M, (frame1.shape[1], frame1.shape[0]))

        # Apply the affine transformation to the second frame
        morphed_frame2 = cv2.warpAffine(frame2, M, (frame2.shape[1], frame2.shape[0]))

        # Blend the morphed frames using the interpolation coefficient
        interpolated_frame = (1 - alpha) * morphed_frame1 + alpha * morphed_frame2

        # Add the interpolated frame to the output array
        interpolated_frames[i] = interpolated_frame

    return interpolated_frames
def super_smooth_interpolate_frames(frame1, frame2, target_frame_num):
    # Convert the input frames to grayscale for edge detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Find the edges in each frame using the Canny edge detector
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)

    # Define a set of corresponding points for the two input frames
    points1 = np.array([[0, 0], [0, frame1.shape[0] - 1], [frame1.shape[1] - 1, frame1.shape[0] - 1]], dtype=np.float32)
    points2 = np.array([[0, 0], [0, frame2.shape[0] - 1], [frame2.shape[1] - 1, frame2.shape[0] - 1]], dtype=np.float32)

    # Calculate the number of frames to interpolate between the two input frames
    num_frames = abs(target_frame_num - 1)

    # Initialize an array to store the interpolated frames
    interpolated_frames = []

    # Interpolate the frames by morphing the shapes of the input frames
    for i in range(num_frames):
        # Calculate the interpolation coefficient
        alpha = (i + 1) / (num_frames + 1)

        # Interpolate the corresponding points
        morphed_points = (1 - alpha) * points1 + alpha * points2

        # Calculate the affine transformation matrix from the corresponding points
        M = cv2.getAffineTransform(points1, morphed_points)

        # Apply the affine transformation to the edges of the first frame
        morphed_edges1 = cv2.warpAffine(edges1, M, (frame1.shape[1], frame1.shape[0]))

        # Apply the affine transformation to the edges of the second frame
        morphed_edges2 = cv2.warpAffine(edges2, M, (frame2.shape[1], frame2.shape[0]))

        # Blend the two edge maps using the interpolation coefficient
        blended_edges = (1 - alpha) * morphed_edges1 + alpha * morphed_edges2

        # Normalize the blended edge map
        blended_edges = cv2.normalize(blended_edges, None, 0, 255, cv2.NORM_MINMAX)

        # Threshold the blended edge map to create a mask
        _, mask = cv2.threshold(blended_edges.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Convert the mask to a color image
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Use the mask to interpolate the color image
        interpolated_frame = (1 - alpha) * frame1 + alpha * frame2
        # Resize the mask to match the interpolated frame
        mask = cv2.resize(mask, (interpolated_frame.shape[1], interpolated_frame.shape[0]))

        # Apply the mask to the interpolated frame
        interpolated_frame = cv2.bitwise_and(interpolated_frame, mask) + cv2.bitwise_and(frame2, cv2.bitwise_not(mask))

        # Add the interpolated frame to the output array
        interpolated_frames.append(interpolated_frame)

    return interpolated_frames
def super_smooth_interpolate_frames_2(frame1, frame2, target_frame_num):
    # Convert the input frames to grayscale for edge detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Find the edges in each frame using the Canny edge detector
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)

    # Define a set of corresponding points for the two input frames
    points1 = np.array([[0, 0], [0, frame1.shape[0] - 1], [frame1.shape[1] - 1, frame1.shape[0] - 1]], dtype=np.float32)
    points2 = np.array([[0, 0], [0, frame2.shape[0] - 1], [frame2.shape[1] - 1, frame2.shape[0] - 1]], dtype=np.float32)

    # Calculate the number of frames to interpolate between the two input frames
    num_frames = abs(target_frame_num - 1)

    # Initialize an array to store the interpolated frames
    interpolated_frames = []
    # Interpolate the frames by morphing the shapes of the input frames
    for i in range(num_frames):
        # Calculate the interpolation coefficient
        alpha = (i + 1) / (num_frames + 1)

        # Interpolate the corresponding points
        morphed_points = (1 - alpha) * points1 + alpha * points2
        print(points1.shape, morphed_points.shape)
        # Calculate the affine transformation matrix from the corresponding points
        M = cv2.getAffineTransform(points1[:3], morphed_points[:3])

        # Apply the affine transformation to the edges of the first frame
        morphed_edges1 = cv2.warpAffine(edges1, M, (frame1.shape[1], frame1.shape[0]))

        # Apply the affine transformation to the edges of the second frame
        morphed_edges2 = cv2.warpAffine(edges2, M, (frame2.shape[1], frame2.shape[0]))

        # Blend the morphed edges using the interpolation coefficient
        blended_edges = (1 - alpha) * morphed_edges1 + alpha * morphed_edges2

        # Convert the blended edges to a grayscale image
        blended_edges = blended_edges.astype(np.uint8)
        blended_gray = cv2.cvtColor(blended_edges, cv2.COLOR_GRAY2BGR)

        # Use the blended edges to interpolate the color image
        interpolated_frame = (1 - alpha) * frame1 + alpha * frame2

        # Apply the blended edges as a mask to the interpolated image
        interpolated_frame = cv2.addWeighted(interpolated_frame, 1, blended_gray, alpha, 0, dtype=cv2.CV_8U)

        # Add the interpolated frame to the output array
        interpolated_frames.append(interpolated_frame)

    return interpolated_frames
def interpolate_linear(p0, p1, fract_mixing):
    r"""
    Helper function to mix two variables using standard linear interpolation.
    Args:
        p0:
            First tensor / np.ndarray for interpolation
        p1:
            Second tensor / np.ndarray  for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a linear mix between both.
    """
    reconvert_uint8 = False
    if type(p0) is np.ndarray and p0.dtype == 'uint8':
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)

    if type(p1) is np.ndarray and p1.dtype == 'uint8':
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)

    interp = (1 - fract_mixing) * p0 + fract_mixing * p1

    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)

    return interp
