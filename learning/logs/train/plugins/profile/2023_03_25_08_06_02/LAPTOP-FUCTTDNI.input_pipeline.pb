  *	X9���@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�2�&c�?!ܑ��N@)p'�_�?1�\ig�I@:Preprocessing2T
Iterator::Root::ParallelMapV2�0���?!7uL�8(@)�0���?17uL�8(@:Preprocessing2�
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat_Cp\�M�?!ę���$@)�\�C���?1��o�#@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map3m��J��?!��F��2@)��9�ا?1!��h� !@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeata����?!֊m	&�@)%̴�++�?1��8�8�@:Preprocessing2v
?Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[12]::ConcatenatejM�S�?!�r��s�@)�ܵ�|Г?1��
�A@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensor%�����?!0�'��	@)%�����?10�'��	@:Preprocessing2v
?Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate���zܷ�?!��5)�@)Uj�@+�?1��ְ��?:Preprocessing2E
Iterator::Root[%X���?!�����+@)��@I��?1�[����?:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch�5Φ#�{?!^�Wb���?)�5Φ#�{?1^�Wb���?:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip:�����?!�:#��P@)Y4���r?1����u�?:Preprocessing2�
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�l���e?!N��u�Z�?)�l���e?1N��u�Z�?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�N^�e?!�L�N�?)�N^�e?1�L�N�?:Preprocessing2�
OIterator::Root::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlice�"�~�F?!��;�r�?)�"�~�F?1��;�r�?:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensor����Mb@?!���8]�?)����Mb@?1���8]�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.