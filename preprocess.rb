require 'rubygems'
require 'bundler/setup'
require 'pry-byebug'
require 'tensor_stream/opencl'
require 'tensor_stream/utils/freezer'


tf = TensorStream
puts "Preprocess Tensorstream version #{tf.__version__} with OpenCL lib #{TensorStream::Opencl::VERSION}"

file_path = File.join('work', 'annotations.yml')

K = 8 # first convolutional layer output depth
L = 8 # second convolutional layer output depth
M = 12 # third convolutional layer
N = 200 # fully connected layer
PIXEL_DEPTH = 3 # (RGB)
CAR_CLASSES = 196
HEIGHT = 200
WIDTH = 320

x = tf.placeholder(:float32, shape: [nil, HEIGHT, WIDTH, PIXEL_DEPTH])

# correct answers will go here
y_ = tf.placeholder(:float32, shape: [nil, CAR_CLASSES])

# step for variable learning rate
step = tf.placeholder(:int32)

pkeep = tf.placeholder(:float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)

w1 = tf.variable(tf.truncated_normal([6, 6, 3, K], stddev: 0.1))
b1 = tf.variable(tf.ones([K])/10)

w2 = tf.variable(tf.truncated_normal([5, 5, K, L], stddev: 0.1))
b2 = tf.variable(tf.ones([L])/10)

w3 = tf.variable(tf.truncated_normal([4, 4, L, M], stddev: 0.1))
b3 = tf.variable(tf.ones([M])/10)

w4 = tf.variable(tf.truncated_normal([80 * 50 * M * PIXEL_DEPTH, N], stddev: 0.1))
b4 = tf.variable(tf.ones([N])/10)

w5 = tf.variable(tf.truncated_normal([N, CAR_CLASSES], stddev: 0.1))
b5 = tf.variable(tf.ones([CAR_CLASSES])/10)

# The model
stride = 1  # output is 320x200
y1 = tf.nn.relu(tf.nn.conv2d(tf.reshape(x, [-1, HEIGHT, WIDTH, 3]), w1, [1, stride, stride, 1], 'SAME') + b1)
stride = 2  # output is 160x100
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, [1, stride, stride, 1], 'SAME') + b2)
stride = 2  # output is 80x50
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, [1, stride, stride, 1], 'SAME') + b3)

# reshape the output from the third convolution for the fully connected layer
yy = tf.reshape(y3, [-1, 80 * 50 * M * PIXEL_DEPTH])
y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
YY4 = tf.nn.dropout(y4, pkeep)
ylogits = tf.matmul(YY4, w5) + b5

# model
y = tf.nn.softmax(ylogits, name: 'out')

# training step, learning rate = 0.003


# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits: ylogits, labels: y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy =  tf.reduce_mean(tf.cast(is_correct, :float32))

# training step, learning rate = 0.003
lr = 0.0001.t +  tf.train.exponential_decay(0.01, step, 2000, 1/Math::E)
train_step = TensorStream::Train::AdamOptimizer.new(lr).minimize(cross_entropy)

@sess = tf.session(profile_enabled: true)
# Add ops to save and restore all the variables.

init = tf.global_variables_initializer

puts "init variables"
@sess.run(init)

#Setup save and restore
puts "restore model"
model_save_path = "test_models/cars_data_3.0"
saver = tf::Train::Saver.new
saver.restore(@sess, model_save_path)
puts "restore done"

train_annotations = YAML.load_file('work/train_annotations.yml' )
test_annotations = YAML.load_file('work/test_annotations.yml' )

def get_images(collection, sub_folder)
  collection.sample(64).map do |fpath, klass|
    puts fpath
    one_hot = Array.new(196) { 0.0 }
    one_hot[klass] = 1.0
    img = File.open(File.join('work', sub_folder, fpath), 'rb') { |io| io.read }

    [@sess.run(TensorStream.image.decode_jpeg(img)), one_hot]
  end
end

puts "loading images"
all_train_images = get_images(train_annotations, 'train')
all_test_images =  get_images(test_annotations, 'test')

test_images = []
test_classes = []

def sample_images(collection, batch_size = 32)
  images = []
  classes = []
  collection.sample(batch_size).each do |image, klass|
    images << image
    classes << klass
  end

  [images, classes]
end

test_batch_x, test_batch_y = sample_images(all_test_images)
test_data = { x => test_batch_x, y_ => test_batch_y, pkeep => 1.0 }

(0..10001).each do |i|
  # load batch of images and correct answers
  batch_x, batch_y = sample_images(all_train_images)
  train_data = { x => batch_x, y_ => batch_y, step => i, pkeep => 0.75 }

  # train
  @sess.run(train_step, feed_dict: train_data)

  if (i % 10 == 0)
    # result = TensorStream::ReportTool.profile_for(sess)
    # File.write("profile.csv", result.map(&:to_csv).join("\n"))
    # success? add code to print it
    a_train, c_train, l = @sess.run([accuracy, cross_entropy, lr], feed_dict: { x => batch_x, y_ => batch_y, step => i, pkeep => 1.0})
    puts "#{i}: accuracy:#{a_train} loss:#{c_train} (lr:#{l})"
  end

  if (i % 100 == 0)
    # success on test data?
    a_test, c_test = @sess.run([accuracy, cross_entropy], feed_dict: test_data)
    puts("#{i}: ******** test accuracy: #{a_test} test loss: #{c_test}")

    # save current state of the model
    save_path = saver.save(@sess, model_save_path)
  end
end



