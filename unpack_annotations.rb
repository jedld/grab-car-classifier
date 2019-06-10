require 'bundler/setup'

require 'nmatrix'
require 'yaml'
require 'fileutils'
require 'pry-rescue'
require 'pry-byebug'

def unpack_file(fpath, out_path, initial: 1, offset: 6)
  filenames = []
  classes = []

  reader = NMatrix::IO::Matlab::Mat5Reader.new(File.open(fpath, "rb+"))
  data = reader.to_a
  pos = initial
  while pos < data.size - offset
    classes << data[pos + 5].to_ruby.to_a[0]
    filenames << data[pos + 6].real_part.data
    pos += offset
  end

  annotations = classes.zip(filenames).map do |k, fname|
    [fname, k]
  end

  FileUtils.mkdir_p 'work'
  File.write(out_path, annotations.to_yaml)
end

unpack_file(File.join('devkit', 'cars_train_annos.mat'), File.join('work', 'train_annotations.yml'))
unpack_file('cars_test_annos_withlabels.mat', File.join('work', 'test_annotations.yml'))