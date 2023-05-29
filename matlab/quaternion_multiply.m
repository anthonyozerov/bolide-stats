function pq = quaternion_multiply(p,q)

% split out scalar and vector
po=p(1); 
p = p(2:4) ; 
qo=q(1); 
q = q(2:4) ; 

pq(1) = po*qo - dot(p,q) ; 
pq(2:4) = po*q + qo*p + cross(p,q) ; 


return