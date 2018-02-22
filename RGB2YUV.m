 %host image
T=imread('./cordova1/f00013.png');

M=imresize(T,[900 600]);

I=im2double(M);

%figure(1),imshow(I); title('Host Image');
 %separate host image into R G & B components
R=I(:,:,1); %figure(2),imshow (R);

G=I(:,:,2); %figure(3),imshow(G);

B=I(:,:,3); %figure(4),imshow(B);

Y = R *  .299000 + G *  .587000 + B *  .114000;

%figure(5),imshow(Y);

U = R * -.168736 + G * -.331264 + B *  .500000 ;

%figure(6),imshow(U);

V = R *  .500000 + G * -.418688 + B * -.081312;

%figure(7),imshow(V);

YUV = cat(3,Y,U,V);

figure(2),imshow(YUV);

R = Y + 1.4075 * (V - 128);

G = Y - 0.3455 * (U - 128) - (0.7169 * (V - 128));

B =  Y + 1.7790 * (U - 128);

RGB = cat(3,R,G,B);

%figure(3),imshow(RGB);