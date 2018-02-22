% lsd_example.m
% Test LSD algorithm with MATLAB
%% show the image.
im = imread('./images/right-hand-road-marker2.png');
imshow(im);
%% get the start_points and end_points of each straight line use LSD.
% note: input parameter is the path of image, use '/' as file separator.
lines = lsd('./images/right-hand-road-marker2.png');
%% plot the lines.
hold on;
for i = 1:size(lines, 2)
    plot(lines(1:2, i), lines(3:4, i), 'LineWidth', lines(5, i) / 2, 'Color', [1, 0, 0]);
end

%% Run histogram equalization and radon here.
% from radon we'll know there's one lane marker here
% so run classification , conbined two lines become one lane, also
% calculate everage lane width , 37 line segments detected 
% line segment format [x1,x2,y1,y2,lane_width(not used)]

%% Classification
%find complete lines 
%pick first random line segment to start with
num_lane = 1;

% disp('start side lane marker detection ...');
% disp(['lane marker number calculated in radon transform : ', num2str(num_lane)]);
% disp('start lane segment classification ...');

line_start_x = lines(:,6);
% for m = 1:i
    temp_x = find( abs(lines(1,:) - line_start_x(1)) < 5 , 2);
% end


%% Linear Regression for all end points 
% end_pt_1(1,:) = lines(1,:);
% end_pt_1(2,:) = lines(3,:);
% end_pt_2(1,:) = lines(2,:);
% end_pt_2(2,:) = lines(4,:);
% 
% end_pt = zeros(i*2,2);
% end_pt(1:i,:) = end_pt_1';
% end_pt(i+1:2*i,:) = end_pt_2';
% 
% 
% poly_x = end_pt(:,1);
% poly_y = end_pt(:,2);
% % run linear regression here
% scatter(poly_x,poly_y);
% 
% p = polyfit(poly_x,poly_y,2);
% 
% sample_x = 1:1:300;
% sample_y = p(1)*sample_x.^2 + p(2)*sample_x + p(3);
% plot(sample_x,sample_y);










