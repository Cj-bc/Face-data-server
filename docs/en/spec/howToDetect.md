# how to calculate each values

Those informations are written at 2019/07/24.
Might be changed in the future.

# X axis rotation

When we shake our head, heigh of head is getting shorter.
Using this, defining heigh of head facing to front to 1.0, facing to completely side to 0.0. This is the same as `cos` cycle.
According to that, I can calculate **how much degrees does X axis rotated** using `acos`(notice that this is just a `degrees` and we don't know whether it should be positive number or negative one)
Then I can detect whether it facing to upside or downside based on movement of face center position.


# Y axis rotation

When we look side, distance between eyes should be shorter.
I use the same method [#X axis rotation] use with this value.


# X axis rotation

It is obtained from the slope of the line connecting the eyes.
Specifically, we connect one point under each eye, calculate the slope and apply it to `atan`.
However, the accuracy is not good because `dlib` does not calculate the eye position properly even if it rotates in the Z-axis direction.
