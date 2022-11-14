function [migSIG,matrix,x,y,z] = rfbf_3D(SIG,x,y,z,param,varargin)

%RFMIG   Migration of RF signals (a generalized version)
%   MIGSIG = RFMIG(SIG,XI,ZI,PARAM) performs a migration of the
%   signals stored in the array SIG. MIGSIG contains the migrated signals.
%   PARAM is a structure that contains all the parameter values required
%   for the migration (see below for more details).
%
%   [MIGSIG,XI,YI,ZI] = RFMIG(SIG,PARAM) determines the ROI automatically and
%   returns the corresponding point coordinates in XI and ZI.
%
%   ---
%   IMPORTANT NOTES:
%   The migrated signals are calculated at the points specified by (XI,YI,ZI).
%   The XI axis is PARALLEL to the transducer and pointing from element #1
%   to element #N (XI = 0 at the CENTER of the transducer). The ZI axis is
%   PERPENDICULAR to the transducer and pointing downward (ZI = 0 at the
%   level of the transducer, ZI increases as depth increases).
%   ---
%
%   The signals - typically RF signals - in SIG must be acquired using a
%   full wave (planar, circular, parabolic...) configuration as used in
%   ultrafast ultrasound imaging. Each column corresponds to a single RF
%   signal over time (as acquired by a single transducer element).
%
%   Note: The code can be excecuted in parallel (the parallel computing
%   toolbox is required). Use <a href="matlab:help matlabpool">MATLABPOOL</a> to enable a parallel job.
%
%   PARAM is a structure that contains the following fields:
%   -------------------------------------------------------
%   1) PARAM.fs: sample frequency (in Hz, REQUIRED)
%   2) PARAM.pitch: pitch of the transducer (in m, REQUIRED)
%   3) PARAM.xn: virtual source position of x direction(in m, REQUIRED)
%      PARAM.yn: virtual source position of y direction(in m, REQUIRED)
%      PARAM.zn: virtual source position of z direction(in m, REQUIRED)
%   4) PARAM.c: longitudinal velocity (in m/s, default = 1540 m/s)
%   5) PARAM.t0: start time for reception (in s, default = 0 s)
%   6) PARAM.method: 1) 'DAS': delay-and-sum (default)
%                    2) 'Kirchhoff': Kirchhoff migration integral (Eq. 5,
%                                    Gazdag 1984, Proceedings of the IEEE)
%
%   NOTE: FKMIG is recommended for the migration of PLANE WAVE data.
%
%   See also EZTXMIG, TXMIG, FKMIG, LUMIG.
%
%   -- Damien Garcia -- 2012/05/01, revised 2013/01/13
%   website: <a
%   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>

assert(ndims(SIG)<=3,['SIG must be a matrix whose each column ',...
    'corresponds to an RF signal acquired by a single element']);
[nz,nx,ny] = size(SIG);

% check if we are in RF or IQ
isIQ = ~isreal(SIG);

if (nargin==5)
elseif (nargin==6)
else
    error('RFMIG must have 5 or 6 input parameters.')
end

%-- Check input parameters
sizx = size(x);
if nargin==4
    assert(isequal(size(z),sizx,size(y)),'X ,Y and Z must be of same size.')
end

if isfield(param,'c0')
    param.c = param.c0;
end
if ~isfield(param,'c')
    warning('Speed of sound is missing. 1540m/s is set by default');
    param.c = 1540; % longitudinal velocity in m/s
end
if ~isfield(param,'fs')
    error('A sampling frequency (PARAM.fs) is required.')
end
if ~isfield(param,'method'), param.method = 'DAS'; end
if strcmpi(param.method,'das')
    isKirchhoff = false;
elseif strcmpi(param.method,'kirchhoff')
    isKirchhoff = true;
else
    error('PARAM.method must be ''DAS'' or ''Kirchhoff''.')
end
if ~isfield(param,'t0')
    param.t0 = 0; % acquisition start time in s
end
if ~isfield(param,'Pitch_x') % in m
    if isfield(param,'dx')
        param.Pitch_x = param.dx; % param.dx was used in the old version
    else
        error('A pitch value (PARAM.Pitch_x) is required.')
    end
end
if ~isfield(param,'Pitch_y') % in m
    if isfield(param,'dy')
        param.Pitch_y = param.dy; % param.dx was used in the old version
    else
        error('A pitch value (PARAM.Pitch_y) is required.')
    end
end


x1 = param.xm;  % Element positions
y1 = param.ym;
fs = param.fs; % sampling frequency
c = param.c; % propagation velocity


if (~isfield(param,'zn') )
    warning('Virtual source is missing. Set a virtual source of (0,0,-10)')
    param.zn = -10;
    param.xn = 0;
    param.yn = 0;
end

if ~isfield(param, 'type')
    param.type = 'TX';
end
if ~isfield(param,'W')
    param.W = ones(nx,ny);
end

if (nargin==5)
    do_summation = 1;
else
    do_summation = 0;
end

% compute the minimal distance from the probe to the virtual source
dmin = min( sqrt( (x1(:)-param.xn).^2 + (y1(:)-param.yn).^2 + param.zn.^2 ) );
if (param.zn<0)
    dmin = -dmin;
end


kk=0;
%-- Migration (diffraction summation)
if (do_summation)
    migSIG = zeros(size(x));
else
    migSIG = zeros([numel(x1) numel(x)]);
end
t0 = param.t0;
SIG = double(SIG);
if isKirchhoff
    % -- Temporal half derivative of the RF signals --
    % see "The Kirchhoff migration integral" in Margrave, 2003
    % see "Migration of seismic data", Gazdag and Sguazzero, 1984
    SIG = halfdiff(SIG);
end


%% Virtual sources
if (isfield('param', 'TXdelay'))
    % rajout des lignes vides...
    [yT, xT] = meshgrid(y1(1):param.Pitch_y:y1(end), x1(:,1));

    %-- Virtual transducer
    TXdelay = param.c * param.TXdelay ;
    F = scatteredInterpolant(x1(:), y1(:), double(TXdelay(:))) ;
    TXdelay = F(xT, yT); % avec l'interpolation

    xTv = xT - diff5_mat(TXdelay.^2,1)/param.Pitch/2;
    yTv = yT - diff5_mat(TXdelay.^2,2)/param.Pitch/2;
    zTv = -sqrt(abs(TXdelay.^2-(xTv-xT).^2-(yTv-yT).^2));

    %%
    subplot(111); plot3(xT(:), yT(:), -TXdelay(:), 'o'); view(15, 45);
    hold on; plot3(xT(:), yT(:), 0*xT(:), '+r'); hold off;
    set(gca, 'zdir', 'reverse');
    hold on; plot3(xTv, yTv, zTv, '*g'); hold off;

    %%
    % a = diff5_mat(TXdelay.^2,1);
    % b = diff5_mat(TXdelay.^2,2);
    % subplot(221); imagesc(TXdelay.^2); subplot(223); imagesc(a); subplot(224); imagesc(b)
    %%
    dTX = zeros(numel(x),1);
    for k=1:numel(x)
        dTX(k) = min(min(sqrt((xTv-x(k)).^2 + (yTv-y(k)).^2 + (zTv-z(k)).^2))); % TX distance
    end
end

%% NEW BY FRANCOIS
S = reshape(SIG, [size(SIG,1) size(SIG,2)*size(SIG,3)]);
if (isIQ)
    wc = 2*pi*param.f0;
end
for k = 1:numel(x1)
    if (~exist('dTX', 'var'))
        dTX = sqrt((x(:)-param.xn).^2+(y(:)-param.yn).^2+(z(:)-param.zn).^2) + dmin; % transmit distance
    end
    dRX = sqrt((x1(k)-x(:)).^2 +(y1(k)-y(:)).^2+ (z(:)).^2); % receipt distance
    if (strcmp(param.type, 'RX'))
        tau = (dRX) / c;
    else
        tau = (dTX + dRX) / c;
    end
    idxt = (tau-t0)*fs + 1;
    I = idxt<1 | idxt>(nz-1);
    idxt(I) = 1; % arbitrary index, will be soon rejected
 
    %-- Interpolation to select the data  for summation
    idx = idxt;
    idxf =  floor(idx);
    
%     TEMP = S(idxf,k).*(idx+1) - S(idxf+1,k).*idx;
    TEMP = S(idxf,k,:).*(idxf+1-idx) + S(idxf+1,k,:).*(idx-idxf); % correction Maxime
    
    if isIQ % phase rotation (if I/Q signals)
        TEMP = TEMP.*exp(1i*wc*tau);
    end
    
    TEMP(I) = 0;
    
    %- obliquity factor  
%     TEMP = reshape( TEMP .* z(:)./(dRX+eps) .* param.W(k)./(nnz(I)+eps), size(x) );
    %- no obliquity factor  
    %TEMP = reshape( TEMP .* param.W(k)./(nnz(I)+eps), size(x) ); 
    TEMP = reshape( TEMP.* param.W(k), size(x) );
    matrix{k}=TEMP;
    if (do_summation)
        migSIG = migSIG + TEMP ;
    else
        migSIG(k,:) = TEMP(:);
    end
        
end
if (~do_summation)
    migSIG = reshape(migSIG, [numel(x1) size(x)]);
end

end

function dy = my_diff(y)
dy = y;
y = y(:);
dy(1) = [-54 13 40 27 -26]*y(1:5)/70;
dy(2) = [-34 3 20 17 -6]*y(1:5)/70;
dy(end-1) = [6 -17 -20 -3 34]*y(end-4:end)/70;
dy(end) = [26 -27 -40 -13 54]*y(end-4:end)/70;
dy(3:end-2) = conv(y,[2 1 0 -1 -2],'valid')/10;
end

function rf = halfdiff(rf)
% See Eq.5 p 1306, Gazdag and Sguazzero, 1984
n0 = size(rf,1);
rf = fft(rf);
f = zeros(1,n0)';
k = 0:n0-1;
I = k<n0/2; f(I) = k(I)/n0; 
I = k>n0/2; f(I) = (k(I)-n0)/n0;

% Note: 1st derivative: rf = bsxfun(@times,rf,2*pi*1i*f);
% Half derivative
rf = bsxfun(@times,rf,sqrt(2*pi*1i*f));

rf = ifft(rf,'symmetric');
end



function dy = diff5(y)
% Differentiation using a five-point stencil:
% LS quadratic fitting on 5 successive points
dy = y;
y = y(:);
dy(1) = [-54 13 40 27 -26]*y(1:5)/70;
dy(2) = [-34 3 20 17 -6]*y(1:5)/70;
dy(end-1) = [6 -17 -20 -3 34]*y(end-4:end)/70;
dy(end) = [26 -27 -40 -13 54]*y(end-4:end)/70;
dy(3:end-2) = conv(y,[2 1 0 -1 -2],'valid')/10;
end



function dy = diff5_mat(y, dim)
% Differentiation using a five-point stencil:
% LS quadratic fitting on 5 successive points
if (dim==2)
    y = y';
end
dy = y;

dy(1,:) = [-54 13 40 27 -26]*y(1:5,:)/70;
dy(2,:) = [-34 3 20 17 -6]*y(1:5,:)/70;
dy(end-1,:) = [6 -17 -20 -3 34]*y(end-4:end,:)/70;
dy(end,:) = [26 -27 -40 -13 54]*y(end-4:end,:)/70;
dy(3:end-2,:) = conv2(y,[2 1 0 -1 -2]','valid')/10;

if (dim==2)
    dy = dy';
end

end
