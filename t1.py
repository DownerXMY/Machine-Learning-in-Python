from flask import Flask,send_from_directory

app = Flask(__name__)
@app.route("/download")
def index():
	return send_from_directory('/Users/mingyuexu/PycharmProjects/',filename="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",as_attachment=True)

@app.route("/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
def func1():
    return send_from_directory('/Users/mingyuexu/PycharmProjects/',filename="resnet50_weights_tf_dim_ordering_tf_kernels_notop(1).h5", as_attachment=True)

@app.route("/NASNet-mobile-no-top.h5")
def func2():
    return send_from_directory('/Users/mingyuexu/PycharmProjects/',filename="NASNet_mobile_notop.h5", as_attachment=True)
if __name__ =='__main__':
    app.run(host='0.0.0.0',port=8080)
