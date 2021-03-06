close all;
clear all;
clc;

%% 
I = imread('mark_test_img.png');
I_resized = imresize(I,[200 300]);
imshow(I_resized)
I_resized_gray = rgb2gray(I_resized);
%BW = edge(I_resized_gray,'Sobel');
%% apply radon transform
theta = 0:180;
figure(2)
[R,xp] = radon(I_resized_gray,theta);
imagesc(theta,xp,R);
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)');
ylabel('X\prime');
set(gca,'XTick',0:20:180);
colormap(hot);
colorbar
%end