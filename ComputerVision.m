% Detect interest points and mark their locations
close all
clear all
format short
format compact

%img = '/home/tsmorz/Pictures/Still Shots/Horseshoe Bend.jpg';
img = 'Small Bend.jpg';
img = imread(img);
%img = imresize(img, 0.1);
%imwrite(img,'Small Bend.jpg')
I = rgb2gray(img);
points = detectSURFFeatures(I);
imshow(I);
hold on;
plot(points.selectStrongest(100));


I1 = im2gray(imread('b1.JPG'));
I2 = im2gray(imread('b3.JPG'));


% Find the corners.
points1 = detectHarrisFeatures(I1);
points2 = detectHarrisFeatures(I2);

% Extract the neighborhood features.
[features1,valid_points1] = extractFeatures(I1,points1);
[features2,valid_points2] = extractFeatures(I2,points2);

% Match the features.
indexPairs = matchFeatures(features1,features2);

% Retrieve the locations of the corresponding points for each image.
matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

% Visualize the corresponding points. You can see the effect of translation between the two images despite several erroneous matches.
figure; 
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);