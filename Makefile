ifndef MODEL
override MODEL = model
endif

compile_model:
	cp ./ipynb/${MODEL}/tf_${MODEL}_MNIST.pbtxt ./mnist_calc_test
	cd mnist_calc_test && mo_tf.py --input_model tf_${MODEL}_MNIST.pbtxt --extensions ./user_mo_extensions \
	--output_dir . --input_shape [1,784] --data_type FP16 --input_model_is_text

compile_opencl:
	cd opencl && /opt/intel/openvino/deployment_tools/tools/cl_compiler/bin/clc \
	--strip-binary-header antirectifier_kernel.cl -o antirectifier_kernel.bin

setup_inference:
	mv ./mnist_calc_test/tf_${MODEL}_MNIST.xml ./mnist_calc_test/mnist_inference.xml 
	mv ./mnist_calc_test/tf_${MODEL}_MNIST.bin ./mnist_calc_test/mnist_inference.bin 

clean:
	rm /tmp/mvnc.mutex

# If you clean by mistake the initial sample model, you can find a spare copy in the 'mnist_calc_test/bak' folder
dist_clean:
	rm -f mnist_calc_test/tf_${MODEL}_*.*
	rm -f opencl/*.bin*
	rm /tmp/mvnc.mutex

devices:
	python3 /opt/intel/openvino/inference_engine/samples/python/hello_query_device/hello_query_device.py

run:
	cd mnist_calc_test && make run