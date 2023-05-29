% DCM to quaternion
% See wikipedia Quaternions and spatial rotation  and
% See wikipedia Rotation representation

function q=dcm2quaternion(M)

% set permuatation uvw of xyz (xyz, yzx, or zxy) 
% so r will have Muu with the largest absolute value
d=diag(M);
i=find(abs(d)==max(abs(d)));
j=circshift([1 2 3]',1-i); 
r=sqrt(1+d(j(1))-d(j(2))-d(j(3))); 

q = [1 0 0 0]; % pre-allocate q to the identity quaternion if r==0 ; 
if r~=0 ; 
    q(1)=0.5*( M(j(3),j(2)) - M(j(2),j(3)) )/r ;
    q(1+j(1)) = r/2 ;
    q(1+j(2)) = 0.5*( M(j(1),j(2)) + M(j(2),j(1)) )/r ;
    q(1+j(3)) = 0.5*( M(j(3),j(1)) + M(j(1),j(3)) )/r ;
end


% Alternative Method use eigenvalues and eigenvectors to find axis and angle
% See wikipedia Rotation representation
%      WARNING: ACOS CANNOT DISTINGUISH THETA FROM -THETA - YOU MAY GET THE
%      CONJUGATE OF THE QUATERNION YOU WANTED
%[v,d]=eig(M); 
%f=abs(1-diag(d)); i=find(f==min(f)); v=v(:,i);% find vector corresponding to eigenvalue closest to 1
%j=[2 3 1]; theta=acos(real(d(j(i),j(i)))) ; % acos real part of any other eigenvalue
%q=anglevector2quat(theta,v);


end