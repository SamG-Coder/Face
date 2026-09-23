# Face construction research applied to the generator

This project should not invent a face by stacking arbitrary smooth bumps on an ellipsoid. The generator now follows the same order used in portrait drawing and sculptural blockout: establish the large head mass, place proportional landmarks, carve major planes, construct features from simple solids, then add smaller soft-tissue forms.

The proportions below are construction defaults, not biological rules. Identity controls are allowed to depart from them.

## 1. Start with the large mass and cut the side planes

Anatomy for Sculptors describes a "helmet head" approach: begin with a ball-like head mass and slice the sides to establish the temple/cheek planes before adding features.

Reference:
- https://anatomy4sculptors.com/blog/forming-the-head-simple-to-complex/

Applied in:
- `kernels/face.cu`: temple, cheek, jaw and chin bands now change the global silhouette instead of only deforming front-facing vertices.
- `src/construction.js`: the 2D front/profile views now show the side-plane cut as part of the blockout.

## 2. Place proportions before features

Useful construction relationships from the Loomis/portrait-drawing material:

- crown-to-hairline is roughly the top sixth, then the visible face can be divided approximately into hairline/brow/nose/chin thirds;
- the eye line is near the vertical midpoint of the head;
- a front-view head is commonly constructed at roughly five eye widths, with about one eye width between the eyes;
- the nose width begins from the inner-eye-corner relationship;
- the mouth corners are usefully related to the pupils/eye centers.

References:
- https://www.proko.com/course-lesson/how-to-draw-the-head-front-view
- https://www.proko.com/course-lesson/how-to-draw-the-head-from-any-angle
- https://anatomy4sculptors.com/blog/forming-the-head-simple-to-complex/

Applied in:
- shared construction landmarks in `face.cu` and `construction.js`;
- eye width/spacing establish the scaffold first;
- nose-wing width is derived from the inner eye corners before `noseWidth` variation;
- mouth width is derived from the eye-center/pupil scaffold before `mouthWidth` variation.

## 3. Eye socket first; eyeball belongs inside it

Portrait construction treats the brow ridge as a block/awning, the socket as the large containing shape, and the eyeball as a sphere sitting deep inside it. Eyelids wrap that sphere and should not be treated as a flat almond pasted onto the face.

References:
- https://www.proko.com/course-lesson/how-to-draw-eyes-anatomy-and-structure
- https://www.proko.com/course-lesson/focus-on-feature-the-eye

Applied in:
- sockets are now compact large-form cavities;
- brow blocks and the central glabella are separate forms;
- the socket is no longer used as a continuation of the nose bridge.

A later pass should add explicit eyeball and eyelid geometry.

## 4. Construct the nose as planes, not a vertical ridge

Proko's nose construction starts with a simple box/wedge: top plane, broad side planes, and bottom plane. The glabella connects the brow region to the nose, while the nasion/saddle is a distinct recess. The bridge itself is bone/cartilage that transitions into the ball and wings; the bridge changes are generally much subtler than a single tall central extrusion.

References:
- https://www.proko.com/course-lesson/how-to-draw-a-nose-anatomy-and-structure
- https://www.proko.com/course-lesson/how-to-draw-a-nose-step-by-step/
- https://www.proko.com/course-lesson/shading-after-construction

Applied in `face.cu`:
- the old long Gaussian `bridge` field was removed;
- a separate glabella form is followed by an explicit nasion recess;
- the dorsum is a tapered wedge with a narrow top plane and much broader side planes;
- projection grows gradually from the saddle toward the lower dorsum;
- tip/ball and alar wings are separate forms;
- a shallow subnasal break is separate from the tip.

This directly addresses the previous "forehead-to-tip pillar" result.

## 5. Put the mouth on a denture/tooth-cylinder mass

The lips sit on a curved tooth-cylinder / muzzle rather than on a flat facial plane. Proko describes the front of the lower face as a staircase of changing planes, with three upper-lip forms and two lower-lip forms.

References:
- https://www.proko.com/course-lesson/how-to-draw-lips-anatomy-and-structure
- https://www.proko.com/lesson/painting-the-head-from-imagination-lighting-without-reference-with-marco-bucci

Applied in `face.cu`:
- a broad denture/muzzle mass is established first;
- the philtrum is recessed into that mass;
- upper lip is split into center + two side forms;
- lower lip is split into two forms;
- the mouth crease curves around the cylinder instead of being a horizontal Gaussian stripe;
- the labiomental groove and chin are separate lower-face steps.

## 6. Preserve blockout planes before smoothing

Anatomy for Sculptors' simple-to-complex workflow explicitly builds the first-level blockout before smoothing transitions into organic anatomy.

Reference:
- https://anatomy4sculptors.com/blog/forming-the-head-simple-to-complex/

The generator therefore uses compact influence fields for feature forms. These reach zero outside their intended region instead of every feature being an infinitely spreading Gaussian. The `Planes` render mode remains useful for checking whether the large construction reads before detail is added.

## Next structural pass

The next geometry work should follow the same order rather than adding skin detail:

1. explicit eyeball spheres positioned inside the sockets;
2. upper/lower eyelid patches wrapping those spheres;
3. actual nose underside with septum, wings and nostril undercuts;
4. mouth opening and lip thickness rather than a single-valued face height field;
5. ears placed from the brow-to-nose construction region;
6. only then soften the blockout and add skin/material detail.
