There should be 4 files: main.py, style.py, utils.py, and vgg19.py. There should also be two folders inside as well: images and pre_trained_model. The first step is to download vgg-verydeep-19 from the following website http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat and put it into pre_trained_model. To run the program, you must first put images that you want to combine into the image folder. Then, save this and open the command prompt. Find the folder from the terminal, and once you are in the main folder, type a line in the following format:

python main.py --content <content file> --style <style file> --output <output file>

Example:
python main.py --content images/city.jpg --style images/wave.jpg --output result.jpg

Please note for this to work the following packages must be installed through pip: tensorflow, numpy, scipy, pillow, and matplotlib. If they have been successfully installed, the program should run and stop after 1000 iteration. You can change some of the arguments from the command prompt as well such as number of iterations, content layers, etc. Just add on to the end of the command the parser input. 
Example:
--num_iter 100

Total Example
python main.py --content images/city.jpg --style images/wave.jpg --output result.jpg --num_iter 100

This lets you change multiple arguments from the command prompt without changing the code. All extra arguments can be found in the file main.py
