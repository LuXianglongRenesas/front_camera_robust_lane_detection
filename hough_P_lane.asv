%% Probabilistic Hough Transform Lane Segment Detection
   % for curve detection 
   
% Xianglong Lu
% Company: Renesas Electonics America
% Aug 2017

%read image
%I = imread('right-hand-road-marker.bmp');

%%
close all
clear all
clc
%
I = imread('Matt_1.png');
%color filtering
hsvImage = rgb2hsv(I);
h = hsvImage(:,:,1);
s = hsvImage(:,:,2);
v = hsvImage(:,:,3);
whitePixels = v > 0.75;
%imshow(whitePixels)

I_ROI_color = imcrop(I,[0 200 1400 600]);
I_ROI = imcrop(whitePixels,[0 200 1400 600]);

%edge detection
BW = edge(I_ROI,'Sobel');
imshow(BW)

%Hough_P processing
[H,T,R] = hough(BW);
% imshow(H,[],'XData',T,'YData',R,...
%             'InitialMagnification','fit');
% xlabel('\theta'), ylabel('\rho');
% axis on, axis normal, hold on;

%find peaks
P  = houghpeaks(H,15,'threshold',ceil(0.2*max(H(:))));

x = T(P(:,2)); y = R(P(:,1));
%plot(x,y,'s','color','white');

%find lines and plot them
lines = houghlines(BW,T,R,P,'FillGap',2,'MinLength',2);
figure, imshow(I_ROI_color), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
%find longest line and highlight it 
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');


















