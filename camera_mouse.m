clear
clc

import java.awt.Robot;
mouse = Robot;
screenSize = get(0, 'screensize');
ScrCent = screenSize(3:4)/2;

faceDetector = vision.CascadeObjectDetector();

% create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% create the webcam object.
cam = webcam();

% capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

%frameSize(2)  = frameSize(2) /2;
%frameSize(1)  = frameSize(1) /2;

Camcent = frameSize(1:2)/2;
% create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2)/2, frameSize(1)/2]+30]);

runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop 

    % get next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    if numPts < 10
        % detection mode.
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            % find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            oldPoints = xyPoints;

            
            bboxPoints = bbox2points(bbox(1, :));

            % convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
       
            
            bboxPolygon = reshape(bboxPoints', 1, []);

            % display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
        % tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % estimate the geometric transformation between the old points
            % and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
     
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % Reset the points.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

    end
            FaceCent = (bboxPoints(1,:)+bboxPoints(2,:)+bboxPoints(3,:)+bboxPoints(4,:))/4;
            movx = double(fix(( Camcent(2)- FaceCent(1))*7));
            movy = double(fix((FaceCent(2) - Camcent(1))*7));
            xx = ScrCent(1)+movx;
            yy = ScrCent(2)+movy;
            mouse.mouseMove(xx, yy);
            if(yy>=screenSize(4)-200)
                mouse.mouseWheel(1);
            end
            if(yy<=200)
                mouse.mouseWheel(-1);
            end
            pause(0.00001);
            
    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);