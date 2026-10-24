function rxSignal = apply_multipath_channel(txSignal, h)
% APPLY_MULTIPATH_CHANNEL
% Apply a discrete-time multipath channel.
% Inputs:
% txSignal - Transmitted signal
% h - Channel impulse response
% Output:
% rxSignal - Received signal after multipath propagation

% Ensure column vectors
txSignal = txSignal(:);
h = h(:);

% Apply multipath channel
rxSignal = conv(txSignal, h);

end