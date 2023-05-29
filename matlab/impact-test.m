function lats=get_lat(N,Vinf,inclination, eclipticLon, tilt)
% tilt is Earth's obliquity in radians

mu = 398600 ; % km3 s-2 standard gravitational parameter of Earth (GM)
R = 6371 ; % km mean radius
Vesc = sqrt(2*mu/R) ; % force minimum impact speed to be escape velocity
% tilt = (23 + 26/60 + 21.448/3600)*pi/180; %radians Earth tilt

M = [1 0 0; 0 cos(tilt) -sin(tilt); 0 sin(tilt) cos(tilt)];
qec2eq = quaternion_conjugate(dcm2quaternion(M));

lats = zeros(N,1);
for i=1:N
    Vr = sqrt(Vinf.^2 + Vesc.^2) ;  % km s-1 speed when impacts Earth
    bmax = R*Vr/Vinf ; % km max impact parameter
    b=sqrt(random('Uniform',0,bmax.^2)) ; % km random radius within max
    % elevation angle of impact
    el = 180/pi*acos(b/R*Vinf/Vr);% deg   acos(b/R*sqrt(1-2*mu/R/(Vr^2))) ; 

    % radial impact parameter when hit Earth
    a = mu/(Vinf^2) ; % km semi-major axis of trajectory relative to Earth
    e = sqrt(1 + (b/a)^2) ; % eccentricity of hyperbola
    f = acos( (b^2/a/R - 1)/e )  ; % rad  true anomaly  = acos( ((a*(e^2-1)/R)-1)/e ) ;
    theta = acos(1/e) ; % half-angle of asymptotes
    psi = pi - f - theta ; 
    z = sin(psi) ; % in planet radii
    % impact point on sphere
    th = random('Uniform',-pi,pi) ; % angle of orientation of impact parameter 
    x=z*cos(th); z=z*sin(th); % impact parameters in planet radii
    y=-sqrt(1-(x^2+z^2)); X=[x,y,z]; %R=1;
    % rotate to inclination
    q=anglevector2quat(inclination,[1 0 0]); 
    X=quaternion_rotate(q,X);  
    % rotate to random angle relative to tilt direction of Earth
    seasonangle = eclipticLon-pi/2; % angle in ecliptic relative to direction of Earth tilt
    q=anglevector2quat(seasonangle,[0 0 1]); 
    X=quaternion_rotate(q,X);  % This is the impact location in ecliptic coords
    % rotate from ecliptic into equatorial frame
    X=quaternion_rotate(qec2eq,X); 
    % rotate by random hour angle to lon/lat
    hour = random('Uniform',-pi,pi) ; % rotation of Earth on axis
    q=anglevector2quat(hour,[0 0 1]);
    X=quaternion_rotate(q,X); 
    % find lon/lat of hit
    [lon,lat]=cart2sph(X(1),X(2),X(3));
    lats(i) = lat;
end

return
