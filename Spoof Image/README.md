In cybersecurity, the term image spoofing refers to an advanced form of visual forgery attack used to deceive identity recognition systems. In these attacks, attackers use 2D images or videos (such as printed photos or images displayed on a screen) or even 3D masks and models to attempt to deceive biometric systems, including facial recognition, thereby allowing unauthorized access to devices or sensitive information.
To combat this threat, advanced technologies are being used. The most important of these is a set called liveness detection, a system that analyzes natural facial movements such as blinking, smiling, or subtle skin changes (detecting color texture and depth) to try to determine whether the image actually belongs to a living person. Both active detection methods (asking the user to move) and passive detection (automatic image analysis without user intervention) are used. Also, deep feature analysis using convolutional neural networks such as MobileNetV2 or Vision Transformer has shown that diagnoses can be made with an accuracy of over 90%.

Structurally, image spoofing falls into two categories:

1. 2D presentation attacks: using still images or videos that are projected onto a screen.
   
2. 3D presentation attacks: using masks, 3D sculptures, or even robots that dynamically mimic faces.


Other countermeasures include using a 3D camera to analyze the depth of the image and examining the reflection of light with an active flash, which enables the detection of differences between the real and simulated facial composition.
As a result, image spoofing is not only a complex technical problem but also a major challenge in digital identity security and combating fraud in authentication processes. Secure systems must use a combination of motion detection, texture and color analysis, depth sensing, and artificial intelligence to ensure image authenticity, so that biometric recognition technology cannot be easily fooled.

"CelebA Database"

With the increasing spread of face-based interactive systems, the security and reliability of these systems have become an important issue, and many research efforts have been made in this field. Among these efforts, countering face spoofing attacks has emerged as an important area, which aims to detect whether the face presented to the system is real or fake.
Despite the progress made in this field, many existing methods still face difficulties in countering complex attacks and in real scenarios. The main reason for this weakness is the limitation of the existing datasets in the number of samples and their diversity. To overcome these challenges, a large dataset in the field of countering face spoofing called CelebA-Spoof was presented in 2020. This dataset contains 625,537 images of 10,177 different individuals, which is much larger in size than previous datasets. The fake images in this set were collected in 8 different situations (a combination of 2 environments × 4 lighting conditions) and using more than 10 different sensors.
<img width="974" height="431" alt="image" src="https://github.com/user-attachments/assets/7375ed30-f658-4dfb-9a88-dfd74e11ae4d" />
