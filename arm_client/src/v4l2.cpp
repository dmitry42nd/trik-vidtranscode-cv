#include "internal/v4l2.h"

#include <QDebug>
#include <QByteArray>
#include <QVector>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <linux/videodev2.h>
#include <libv4l2.h>



namespace trik
{

namespace // <unnamed>
{

class V4L2BufferMapperMemoryMmap : public V4L2BufferMapper
{
  public:
    explicit V4L2BufferMapperMemoryMmap(size_t _count) :m_requestedCount(_count), m_buffers() {}

    virtual bool map(V4L2* _v4l2)
    {
      v4l2_requestbuffers requestBuffers = v4l2_requestbuffers();
      requestBuffers.count = m_requestedCount;
      requestBuffers.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      requestBuffers.memory = V4L2_MEMORY_MMAP;

      if (!v4l2_ioctl(_v4l2, VIDIOC_REQBUFS, &requestBuffers))
        return false;

      m_buffers.resize(requestBuffers.count);
      for (ssize_t idx = 0; idx < m_buffers.size(); ++idx)
      {
        v4l2_buffer buffer = v4l2_buffer();
        buffer.index = idx;
        buffer.type = requestBuffers.type;
        buffer.memory = requestBuffers.memory;

        if (!v4l2_ioctl(_v4l2, VIDIOC_QUERYBUF, &buffer))
        {
          m_buffers.clear();
          qWarning() << "V4L2 query buffer failed";
          return false;
        }

        m_buffers[idx] = QSharedPointer<MmapEntry>(new MmapEntry(v4l2_fd(_v4l2), buffer.length, buffer.m.offset));
      }

      return true;
    }

    virtual bool unmap(V4L2* _v4l2)
    {
      m_buffers.clear();
      return true;
    }

  private:
    class MmapEntry
    {
      public:
        explicit MmapEntry(int _fd, size_t _size, off_t _offset)
         :m_ptr(MAP_FAILED),
          m_size(_size)
        {
          m_ptr = v4l2_mmap(NULL, m_size, PROT_READ|PROT_WRITE, MAP_SHARED, _fd, _offset);
          if (m_ptr == MAP_FAILED)
          {
            qWarning() << "v4l2_mmap failed:" << errno;
            m_size = 0;
            return;
          }
        }

        ~MmapEntry()
        {
          if (m_ptr != MAP_FAILED)
            v4l2_munmap(m_ptr, m_size);
        }

        void* ptr() const
        {
          return m_ptr;
        }

        size_t size() const
        {
          return m_size;
        }

      private:
        void*  m_ptr;
        size_t m_size;
    };

    size_t m_requestedCount;
    QVector<QSharedPointer<MmapEntry> > m_buffers;
};

} // namespace <unnamed>




class V4L2::FdHandle
{
  public:
    explicit FdHandle(const QString& _path)
     :m_fd(-1),
      m_opened(false)
    {
      QByteArray path(_path.toLocal8Bit()); // pin temporary, convert to char*

      m_fd = v4l2_open(path.data(), O_RDWR|O_NONBLOCK, 0);
      if (m_fd < 0)
      {
        qWarning() << "v4l2_open(" << _path << ") failed:" << errno;
        return;
      }

      m_opened = true;
    }

    ~FdHandle()
    {
      if (m_fd)
        v4l2_close(m_fd);
    }

    bool opened() const
    {
      return m_opened;
    }

    int fd() const
    {
      return m_fd;
    }

    bool ioctl(int _request, void* _argp, bool _reportError, int* _errno)
    {
      if (!m_opened)
      {
        if (_errno)
          *_errno = ENOTCONN;
        if (_reportError)
          qWarning() << "v4l2_ioctl(" << _request << ") attempt on closed device";
        return false;
      }

      if (v4l2_ioctl(m_fd, _request, _argp) == -1)
      {
        if (_errno)
          *_errno = errno;
        if (_reportError)
          qWarning() << "v4l2_ioctl(" << _request << ") failed:" << errno;
        return false;
      }

      if (_errno)
        *_errno = 0;

      return true;
    }

  private:
    int  m_fd;
    bool m_opened;

    FdHandle(const FdHandle&) = delete;
    FdHandle& operator=(const FdHandle&) = delete;
};




class V4L2::FormatHandler
{
  public:
    explicit FormatHandler(V4L2* _v4l2, const ImageFormat& _imageFormat)
     :m_v4l2(_v4l2),
      m_opened(false),
      m_v4l2Format(),
      m_imageFormat()
    {
      if (!setFormat(_imageFormat))
        return;

      if (!reportEmulatedFormats())
        return;

      if (!fetchFormat())
        return;

      m_opened = true;
    }

    bool opened() const
    {
      return m_opened;
    }

    const ImageFormat& imageFormat() const
    {
      return m_imageFormat;
    }

  private:
    V4L2*        m_v4l2;
    bool         m_opened;
    v4l2_format  m_v4l2Format;
    ImageFormat  m_imageFormat;

    bool setFormat(const ImageFormat& _imageFormat)
    {
      m_v4l2Format = v4l2_format();
      m_v4l2Format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      m_v4l2Format.fmt.pix.width       = _imageFormat.m_width;
      m_v4l2Format.fmt.pix.height      = _imageFormat.m_height;
      m_v4l2Format.fmt.pix.pixelformat = _imageFormat.m_format ? _imageFormat.m_format.id() : V4L2_PIX_FMT_YUYV;
      m_v4l2Format.fmt.pix.field       = V4L2_FIELD_NONE;

      if (!m_v4l2->fd_ioctl(VIDIOC_S_FMT, &m_v4l2Format))
        return false;

      return true;
    }

    bool fetchFormat()
    {
      m_imageFormat.m_width      = m_v4l2Format.fmt.pix.width;
      m_imageFormat.m_height     = m_v4l2Format.fmt.pix.height;
      m_imageFormat.m_format     = FormatID(m_v4l2Format.fmt.pix.pixelformat);
      m_imageFormat.m_lineLength = m_v4l2Format.fmt.pix.bytesperline;
      m_imageFormat.m_size       = m_v4l2Format.fmt.pix.sizeimage;

      return true;
    }

    bool reportEmulatedFormats() const
    {
      for (size_t fmtIdx = 0; ; ++fmtIdx)
      {
        v4l2_fmtdesc fmtDesc;
        fmtDesc.index = fmtIdx;
        fmtDesc.type = m_v4l2Format.type;

        int err;
        if (!m_v4l2->fd_ioctl(VIDIOC_ENUM_FMT, &fmtDesc, false, &err))
        {
          if (err != EINVAL)
          {
            qWarning() << "v4l2_ioctl(VIDIOC_ENUM_FMT) failed:" << err;
            return false;
          }
          break;
        }

        if (fmtDesc.pixelformat == m_v4l2Format.fmt.pix.pixelformat)
        {
          if (fmtDesc.flags & V4L2_FMT_FLAG_EMULATED)
            qWarning() << "V4L2 format" << FormatID(m_v4l2Format.fmt.pix.pixelformat).toString() << "is emulated, performance will be degraded";

          break;
        }
      }
      return true;
    }


    FormatHandler(const FormatHandler&) = delete;
    FormatHandler& operator=(const FormatHandler&) = delete;
};




int
V4L2BufferMapper::v4l2_fd(V4L2* _v4l2)
{
  return _v4l2->fd();
}

bool
V4L2BufferMapper::v4l2_ioctl(V4L2* _v4l2, int _request, void* _argp, bool _reportError, int* _errno)
{
  return _v4l2->fd_ioctl(_request, _argp, _reportError, _errno);
}




class V4L2::BufferMapperWrapper
{
  public:
    explicit BufferMapperWrapper()
     :m_v4l2(),
      m_opened(false),
      m_bufferMapper()
    {
    }

    explicit BufferMapperWrapper(V4L2* _v4l2, const QSharedPointer<V4L2BufferMapper>& _bufferMapper)
     :m_v4l2(_v4l2),
      m_opened(false),
      m_bufferMapper(_bufferMapper)
    {
      if (!m_v4l2 || !m_bufferMapper)
        return;

      if (!m_bufferMapper->map(m_v4l2))
        return;

      m_opened = true;
    }

    ~BufferMapperWrapper()
    {
      if (m_v4l2 && m_bufferMapper && m_opened)
        m_bufferMapper->unmap(m_v4l2);
    }

    bool opened() const
    {
      return m_opened;
    }

  private:
    V4L2* m_v4l2;
    bool  m_opened;
    QSharedPointer<V4L2BufferMapper> m_bufferMapper;
};




V4L2::V4L2(QObject* _parent)
 :QObject(_parent),
  m_fdHandle(),
  m_formatHandler(),
  m_path("/dev/video0"),
  m_formatConfigured(),
  m_formatActual(),
  m_bufferMapperConfigured(new V4L2BufferMapperMemoryMmap(2)),
  m_bufferMapperActual(),
  m_frameNotifier()
{
}

V4L2::~V4L2()
{
}

void
V4L2::setDevicePath(const QString& _path)
{
  m_path = _path;
}

void
V4L2::setFormat(const ImageFormat& _format)
{
  m_formatConfigured = _format;
}

void
V4L2::setBufferMapper(const QSharedPointer<V4L2BufferMapper>& _bufferMapper)
{
  m_bufferMapperConfigured = _bufferMapper;
}

bool
V4L2::open()
{
  if (m_fdHandle)
    return true;

  m_fdHandle.reset(new FdHandle(m_path));
  if (!m_fdHandle->opened())
  {
    qWarning() << "V4L2 open failed";
    m_fdHandle.reset();
    return false;
  }

  m_formatHandler.reset(new FormatHandler(this, m_formatConfigured));
  if (!m_formatHandler->opened())
  {
    qWarning() << "V4L2 format setup failed";
    m_formatHandler.reset();
    m_fdHandle.reset();
    return false;
  }

  emit formatChanged(m_formatHandler->imageFormat());
  emit opened();

  return true;
}

bool
V4L2::close()
{
  stop(); // just in case

  if (!m_fdHandle)
    return true;

  m_formatHandler.reset();
  m_fdHandle.reset();

  emit closed();

  return true;
}

bool
V4L2::start()
{
  open(); // just in case

  if (!m_fdHandle)
  {
    qWarning() << "V4L2 not opened";
    return false;
  }

  if (m_bufferMapperActual)
    return true;

  m_bufferMapperActual.reset(new BufferMapperWrapper(this, m_bufferMapperConfigured));
  if (!m_bufferMapperActual->opened())
  {
    qWarning() << "V4L2 buffers map failed";
    m_bufferMapperActual.reset();
    return false;
  }

  m_frameNotifier.reset(new QSocketNotifier(m_fdHandle->fd(), QSocketNotifier::Read, this));
  connect(m_frameNotifier.data(), SIGNAL(activated(int)), this, SLOT(frameReady()));

  emit started();

  return true;
}

bool
V4L2::stop()
{
  if (!m_bufferMapperActual)
    return true;

  m_frameNotifier.reset();
  m_bufferMapperActual.reset();

  emit stopped();

  return true;
}

void
V4L2::reportFPS()
{
#warning TODO
}

void
V4L2::frameReady()
{
#warning TODO
  qDebug() << __func__;
}

int
V4L2::fd() const
{
  if (!m_fdHandle)
    return -1;

  return m_fdHandle->fd();
}

bool
V4L2::fd_ioctl(int _request, void* _argp, bool _reportError, int* _errno)
{
  if (!m_fdHandle)
  {
    if (_errno)
      *_errno = ENOTCONN;
    if (_reportError)
      qWarning() << "v4l2_ioctl(" << _request << ") attempt on closed device";
    return false;
  }

  return m_fdHandle->ioctl(_request, _argp, _reportError, _errno);
}

} // namespace trik
