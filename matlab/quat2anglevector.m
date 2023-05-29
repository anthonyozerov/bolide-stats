function [angle,vector]=quat2anglevector(quaternion)
%
% Converts a quaternion into a vector and the angle to rotate around it
% [angle,vector]=quat2anglevector(quaternion)

angle = acos(1-2*quaternion(1).^2) ; 
vector = quaternion(2:4)./sin(angle/2) ; 


return