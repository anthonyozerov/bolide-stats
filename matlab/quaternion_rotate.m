function rotatedvector = quaternion_rotate(quaternion,vector)
% function rotatedvector = quaternion_rotate(quaternion,vector)
%
% Rotation of a vector by a quaternion
%
% This is a vector rotation interpretation. 
% Can also be done by considering the vector as a purely imaginary
% quaternion ie v_quat = (0,v1,v2,v3) and then just using quaternion
% multiplication of q* * v_quat * q
%
% To perform multiple rotations at once, q and v must be column matrices
% (4xn) and (3xn)
%   
if numel(vector)==3 % single vector
    qo=quaternion(1); 
    q=quaternion(2:4);
    rotatedvector = (2*qo^2-1)*vector  +  2*dot(vector,q)*q  + 2*qo*cross(vector,q) ;
else 
    % Multiple rotations: q and v must be column matrices (4xn) and (3xn)
    qo=quaternion(1,:); 
    q=quaternion(2:4,:);
    rotatedvector = repmat((2*qo.^2-1),3,1).*vector  ...
                 +  repmat(2*dot(vector,q),3,1).*q   ...
                 + 2*repmat(qo,3,1).*cross(vector,q) ;
end

end