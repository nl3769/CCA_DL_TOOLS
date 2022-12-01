function probe = fct_get_probe(varargin)
    
    switch nargin    
        case 1
            probe = getparam(varargin{1});
        case 2
            nb_elements = varargin{1};
            lambda = varargin{2};
            probe = mk_probe(nb_elements, lambda);
    end
end

% -------------------------------------------------------------------------
function probe = mk_probe(nb_elements, lambda)

    probe.Nelements = nb_elements;
    probe.height = 0.005;
    probe.focus = 10;
    probe.width = lambda * 0.95;
    probe.kerf = lambda * 0.05;
    probe.pitch = lambda;
    probe.radius = Inf;
    
end