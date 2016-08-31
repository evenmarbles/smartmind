from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# noinspection PyUnresolvedReferences
from six.moves.urllib.request import urlopen
# noinspection PyUnresolvedReferences
from six.moves.urllib.error import URLError, HTTPError

import os
import sys
import hashlib
import tarfile

from collections import OrderedDict

import tensorflow as tf

from . import tolist

# Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
# urllib module, known to have issues with proxy management
if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        def chunk_read(response, chunk_size=8192, reporthook=None):
            total_size = response.info().get('Content-Length').strip()
            total_size = int(total_size)
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                count += 1
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def _get_reader(name, params):
    def eval_params(p):
        default = {
            'text': OrderedDict([('skip_header_lines', None), ('name', None)]),
            'whole_file': OrderedDict([('name', None)]),
            'identity': OrderedDict([('name', None)]),
            'tf_record': OrderedDict([('name', None), ('options', None)]),
            'fixed_length': OrderedDict
            ([('record_bytes', None), ('header_bytes', None), ('footer_bytes', None), ('name', None)]),
        }[name]

        p = {} if p is None else p

        if isinstance(p, dict):
            if not all(k in default for k in p.keys()):
                raise Exception('{}: Reader parameter mismatch.'.format(name))
            return p

        p = tolist(p)
        if len(p) > len(default):
            raise Exception('{}: Too many reader parameters given.'.format(name))
        p = dict(zip(default.keys(), p))

        if name == 'fixed_length' and p['record_bytes'] is None:
            raise Exception('{}: Parameter `record_bytes` is required.'.format(name))
        return p

    return {
        'text': tf.TextLineReader,
        'whole_file': tf.WholeFileReader,
        'identity': tf.IdentityReader,
        'tf_record': tf.TFRecordReader,
        'fixed_length': tf.FixedLengthRecordReader
    }[name](**eval_params(params))


def _decode_text(value):
    raise NotImplementedError


def _decode_whole_file(value):
    raise NotImplementedError


def _decode_identity(value):
    raise NotImplementedError


def _decode_tf_record(value):
    raise NotImplementedError


def _decode_fixed_length(value):
    # Convert from a string to a vector of uint8 that is record_bytes long.
    return tf.decode_raw(value, tf.uint8)


def validate_file(filename, md5_hash):
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False


def download_and_extract(dest_dir, data_dir, origin, untar=False, md5_hash=None):
    filename = origin.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not untar:
        data_dir = os.path.join(data_dir, dest_dir)

    download = False
    if os.path.exists(filepath):
        # file found; verify integrity if a hash was provided
        if md5_hash is not None:
            if not validate_file(filepath, md5_hash):
                print('A local file was found, but it seems to be'
                      ' incomplete or outdated.')
                download = True
    else:
        download = True

    if download:
        def dl_progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, filepath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(filepath):
                os.remove(filepath)
            raise

        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if untar:
        untar_dir = os.path.join(data_dir, dest_dir)
        if not os.path.exists(untar_dir):
            print('Untaring file...')
            tfile = tarfile.open(filepath, 'r:gz')
            tfile.extractall(data_dir)
            tfile.close()
        data_dir = untar_dir
    return data_dir


def get_file(dest_dir, origin, untar=False, cache_subdir='datasets'):
    data_dirbase = os.path.expanduser(os.path.join('~', '.smartmind'))
    if not os.access(data_dirbase, os.W_OK):
        data_dirbase = os.path.join('tmp', '.smartmind')
    data_dir = os.path.join(data_dirbase, cache_subdir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = download_and_extract(dest_dir, data_dir, origin, untar=untar)
    return data_dir


def read_data_using_reader_op(filenames, reader='fixed_length', reader_params=None, cb_preprocess=None):
    """Reads examples from data files using the Reader op.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Parameters
    ----------
    filenames: list
    reader: str
    reader_params: dict or list or tuple
    cb_preprocess: callable

    Returns
    -------
    DataRecord:
        An object representing a single example.
    """
    class DataRecord(object):
        pass

    result = DataRecord()

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    reader_str = reader

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read a record, getting filenames from the queue.
    reader = _get_reader(reader_str, reader_params)
    result.key, value = reader.read(filename_queue)

    record_bytes = {
        'text': _decode_text,
        'whole_file': _decode_whole_file,
        'identity': _decode_identity,
        'tf_record': _decode_tf_record,
        'fixed_length': _decode_fixed_length
    }[reader_str](value)

    if cb_preprocess is not None:
        result = cb_preprocess(record_bytes, result)

    return result
