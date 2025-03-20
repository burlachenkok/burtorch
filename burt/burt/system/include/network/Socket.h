/** @file
* Cross-platform implementation of network socket
*/

#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

#if BURT_WINDOWS
#include <winsock2.h>
#include <WS2tcpip.h>

#else
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>

#define SOCKET int
#define SOCKET_ERROR -1
#define INVALID_SOCKET -1
#define SD_RECEIVE SHUT_RD
#define SD_SEND SHUT_WR
#endif

#ifndef SOMAXCONN
    #define SOMAXCONN 128
#endif

#include "burt/copylocal/include/Data.h"
#include "burt/copylocal/include/MutableData.h"

#include <string>
#include <sstream>
#include <memory>
#include <assert.h>

namespace burt
{
    /* @class Socket
    * @brief Neutral Berkeley Sockets for network connection
    */
    class Socket
    {
    public:

        /** Is the initialization of the network subsystem have been performed
        */
        static bool isNetworkSubSystemInitialized();

        /** The initialization of the network subsystem
        */
        static bool initNetworkSubSystem();

        /** The deinitialization of the network subsystem
        */
        static void deinitNetworkSubSystem();

        /** Protocol type for socket. IPV4 containts 4 byte (32 bit) addresses, IPV6 - 16 bytes (128 bit) addresses.
        * See also IP minimum reassembly buffer size: IPv4 minimum MTU which is 576 bytes, and IPv6 minimum MTU is 1500 bytes.
        */
        enum class Protocol
        {
            UNKNOWN, ///< Unknown protocol
            TCPv4,   ///< TCP4. Reliable, Bidirectional, Byte-stream, Connection orientated.
            UDPv4,   ///< UDP4. Messages may arrive out of order, be duplicated, or not arrive at all. Not byte-stream, but datagrams are transfered. Message boudaries are preserved. Connectionalles. In principle 548 bytes of payload garantees absence of "BAD" things as fragmentation.
            TCPv6,   ///< TCP6. Reliable, Bidirectional, Byte-stream, Connection orientated
            UDPv6    ///< UDP6. Messages may arrive out of order, be duplicated, or not arrive at all. Not byte-stream, but datagrams are transfered. Message boudaries are preserved. Connectionalles.
        };

        Socket();

        Socket(Protocol protocol);

        ~Socket();

        Socket(const Socket& rhs) = delete;

        Socket(Socket&& rhs) noexcept;

        Socket& operator = (Socket&& rhs) noexcept;

        /** Get the type of protocol used for the socket
        * @return protocol type
        */
        Protocol getProtocol() const;

        /** Assign (bind) a socket to the port and address
        * @param address address for which we will bind a socket. 
        * @param port port number
        * @param reuseAddress boolean flag that allows to reuse port.
        * @return true, if the operation is completed successfully
        * 
        * @remark Pass NULL or "0.0.0.0" if you want to receive data from any network interface.
        * @remark Usually, a server employs this call to bind its socket to a well-known address so that clients can locate the socket
        * @remark After binding to port socket can be moved in listening mode
        * @remark If an application does not select a particular port then TCP and UDP assign a unique ephemeral (short-lived) port
        * @remark TCP and UDP also assign an ephemeral port number if we bind a socket to port 0
        * @remark After closing TCP endpoint that remains in the TIME_WAIT state until the 2MSL timeout expires. RFC 1122 recommends a value of 2 minutes for the MSL.
        */
        bool bind(const char * address, unsigned short port, bool reuseAddress = false);

        /** Bind to the port and allow obtaining data from any network interface.
        * @param port port number
        * @return true, if the operation is completed successfully
        */
        bool bind(unsigned short port, bool reuseAddress = false);

        /** The function moves a socket in listening mode on the bind port for allow incoming connections
        * @param maxPendingConnections the maximum number of requests that can be stored in the queue
        * @return true if the operation is completed successfully
        * 
        * @remark Datagram(UDP) servers do not call "listen, accept"
        * @remark Datagram(UDP) servers work without establishing a connection and immediately after the binding may be used to read incoming messages.
        *
        * @remark Before move socket in listening mode it should be binded to the port
        * @remark backlog queue is needed because server is busy handling some other client(s) and it does not call yet next accept.
        */
        bool listen(int maxPendingConnections = SOMAXCONN);

        /** Establish connection with the server (another remote socket).
        * @param address ip address of the server or the server DNS name
        * @param port server port
        * @return true if the connection is successfully established
        * 
        * @remark Datagram protocol can be used without connection.
        * @remark For TCP connection is established via sending 3 SIN package
        * @remark In case of problem with establishing connection the connection establishment happens TcpMaxConnectRetransmissions times. At intervals of 3 - 6 - 12 seconds.        
        * 
        * @remark For UDP connection causes the kernel to record a particular address as this sockets peer. The effect of connect() is asymmetric for datagram socket. If you use it such socket is called "connected datagram socket".
        * @remark If connection is failed, the fresh socket should be created and connect should be tried one more time.
        */
        bool connect(const char * address, unsigned short port);

        /** Blocking calls to receive incoming connections for socket in passive (not active) mode
        * @return an incoming connection
        * @remark Extracts the first connection on the queue of pending connections on socket
        * @remark Socket before accepting connection should be moved into listening mode.
        */
        std::unique_ptr<Socket> serverAcceptConnection();

        /** Send the data without waiting from receiver side.
        * @param buf pointer to the buffer data - Power
        * @param len data buffer length
        * @return true, if everything is ok
        * @remark Please use if you understand what you're doing
        */
        bool sendData(const void * buff, int len);

        /** Send a single byte
        * @param val value of byte type
        * @return true, if everything is ok
        */
        bool sendByte(uint8_t val) {
            return sendData(&val, sizeof(val));
        }
        
        template<class Mat,
                 bool putMatrixSize = true,
                 bool putMatrixSizeAsVaryingInteger = true>
        bool sendMatrixItems(const Mat& m) 
        {
            size_t cols = m.columns();
            size_t rowsInBytes = m.rows() * sizeof(typename Mat::TElementType);
            size_t LDA = m.LDA;

            const typename Mat::TElementType* rawData = m.matrixByCols.dataConst();
            
            if constexpr (putMatrixSize)
            {
                if constexpr (putMatrixSizeAsVaryingInteger)
                    sendUnsignedVaryingInteger(cols * rowsInBytes);
                else
                    sendUint64(cols * rowsInBytes);
            }

            for (size_t j = 0, offset = 0; j < cols; ++j, offset += LDA)
            {
                sendData(rawData + offset, rowsInBytes);
            }
            
            return true;
        }

        template<class Mat,
                 bool getMatrixSize = true,
                 bool getMatrixSizeAsVaryingInteger = true>
        bool recvMatrixItems(Mat& m)
        {
            size_t cols = m.columns();
            size_t rowsInBytes = m.rows() * sizeof(typename Mat::TElementType);
            size_t LDA = m.LDA;

            typename Mat::TElementType* rawData = m.matrixByCols.data();
            
            if constexpr (getMatrixSize)
            {
                uint64_t msgSize = 0;

                if constexpr (getMatrixSizeAsVaryingInteger)
                    msgSize = getUnsignedVaryingInteger();
                else
                    msgSize = getUint64();

                assert(msgSize == cols * rowsInBytes);
            }

            for (size_t j = 0, offset = 0; j < cols; ++j, offset += LDA)
            {
                recvData(rawData + offset, rowsInBytes);
            }
            return true;
        }
        
#if 0
        template<class Mat>
        bool sendMatrixItemsWithSize(const Mat& m)
        {
            size_t cols = m.columns();
            size_t rows = m.rows();
            size_t rowsInBytes = rows * sizeof(typename Mat::TElementType);
            size_t LDA = m.LDA;

            const typename Mat::TElementType* rawData = m.matrixByCols.dataConst();
            sendUint64(rows);
            sendUint64(cols);

            for (size_t j = 0, offset = 0; j < cols; ++j, offset += LDA)
            {
                sendData(rawData + offset, rowsInBytes);
            }
            return true;
        }


        template<class Mat>
        bool recvMatrixItemsWithSize(const Mat& m)
        {
            size_t cols = m.columns();
            size_t rows = m.rows();

            uint64_t recv_cols = 0;
            uint64_t recv_rows = 0;

            recvData(&recv_rows, sizeof(recv_cols));
            recvData(&recv_cols, sizeof(recv_cols));

            if (cols != recv_cols || rows != recv_rows)
            {
                m = Mat(recv_rows, recv_cols);
            }
            cols = recv_cols;
            rows = recv_rows;

            size_t rowsInBytes = rows * sizeof(typename Mat::TElementType);
            size_t LDA = m.LDA;
            typename Mat::TElementType* rawData = m.data();

            for (size_t j = 0, offset = 0; j < cols; ++j, offset += LDA)
            {
                recvData(rawData + offset, rowsInBytes);
            }

            return true;
        }
#endif
        
        template<class TIntType>
        bool sendUnsignedVaryingInteger(TIntType value)
        {
            burt::MutableData buffer;
            bool result = buffer.putUnsignedVaryingInteger<TIntType>(value);
            result &= sendData(buffer.getPtr(), buffer.getFilledSize());
            return result;
        }

        template<class TIntType, TIntType value>
        bool sendUnsignedVaryingIntegerKnowAtCompileTime()
        {
            if constexpr (value < 128)
                return sendByte((uint8_t)value);
            else
                return sendUnsignedVaryingInteger<TIntType>(value);
        }
        
        /** Send a single uint64 value
        * @param val value of uint64 type
        * @return true, if everything is ok
        */
        bool sendUint64(uint64_t val) {
            return sendData(&val, sizeof(val));
        }

        /** Send a single uint32 value
        * @param val value of uint32 type
        * @return true, if everything is ok
        */
        bool sendUint32(uint32_t val) {
            return sendData(&val, sizeof(val));
        }

        /** Send a single uint16 value
        * @param val value of uint16 type
        * @return true, if everything is ok
        */
        bool sendUint16(uint16_t val) {
            return sendData(&val, sizeof(val));
        }
      
        /** Get the data with waiting
        * @param buf pointer to the buffer receiver data
        * @param len the length of the buffer data
        * @return true if all is ok
        */
        bool recvData(void * buff, int len);
        
        template<class TResultType = uint64_t>
        TResultType getUnsignedVaryingInteger()
        {
            char buffer[sizeof(TResultType) * 8 / 7 + 1];
            size_t bufferPos;

            static_assert((sizeof(buffer) * 7) / 8 * 8 >= sizeof(TResultType) * 8, "buffer has enough capacity to store any value");

            for (bufferPos = 0; ; ++bufferPos)
            {
                assert(bufferPos < sizeof(buffer));
                bool recvResult = recvData(buffer + bufferPos, 1);
                assert(recvResult == true);
                // last byte for varying integer
                if ((buffer[bufferPos] & 0b1000'0000) == 0)
                    break;
            }
            
            uint64_t result = burt::Data(buffer, 
                                         bufferPos + 1, 
                                         burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree).getUnsignedVaryingInteger();

            return result;
        }
        
        /** Get a single uint64 value from the socket
        */
        uint64_t getUint64();
        
        /** Get a single uint32 value from the socket
        */
        uint32_t getUint32();
        
        /** Get a single uint16 value from the socket
        */
        uint16_t getUint16();

        /** Get a single byte from the socket
        */
        uint8_t getUint8();
            
        /** Set timeout to receive new data
        * @param milliseconds timeout to receive new data
        * @return true if the time was successfully set
        */
        bool setRecvBlockTimeout(unsigned int milliseconds);

        /** Shutdown receive channel
        */
        void shutDownReceiveChannel() const;

        /** Shutdown sending channel
        */
        void shutDownSendChannel() const;

        /** Get underlying OS socket handle
        *@return socket handle from OS
        */
        SOCKET getHandleFromOS() const;

        /** Get text description with ip:port
        * @return text description
        */
        const std::string& getAddressInfo() const;

        /** Get avilable bytes to read from socket
        * @return The number of bytes available to read from socket without blocking
        */
        size_t getAvailableBytesForRead() const;
        
        /** Is socket active
        * @return The status
        * @remark Active sockets can be used to carry connection and send data.
        * @returm true if socket is active
        */
        bool isSocketActive() const;
        
        /** Set some TCP sockets options regaring the communication optimization
        * @param noDelay If set, disable the Nagle algorithm. Segments are always sent as soon as possible. [TCP_NODELAY]
        * @retur true if the option was set
        */
        bool setNoDelay(bool noDelay);
        
        /** Wait for available data to read from a array of sockets
        * @param socketsList pointer to a first element in array of sockets
        * @param dataIsReady4Socket pointer to a first element in array of flags, which will be set to true if data is ready to read from corresponding socket.
        * @param kSockets number of sockets to examine
        * @param microseconds2wait number of microseconds to wait for data to be available.
        * @remark If set microseconds2wait to 0 then the call will not be blocked
        */
        static bool waitForAvailableReadSocket(const Socket* socketsList, 
                                               bool* dataIsReady4Socket, 
                                               size_t kSockets, 
                                               long microseconds2wait);
        
    protected:

        /** Form the structure of the socket address
        * @param ddress string with the domain name OR ipv4 address OR "255.255.255.255" to the broadcast packet OR "0.0.0.0" OR NULL - if you use any address(any network interface card)
        * @param port port number
        * @param error error flag if it is interested in the calling party
        * @return generated address
        */
        sockaddr_in getSocketAddressIPv4(const char * address,
                                       unsigned short port, 
                                       bool * error = NULL) const;

        /** Form the structure of the socket address
        * @param address string with the domain name, or ipv6 address written as 2 bytes x 8 = 16 bytes = 128 bits in HEX format
        * @param port port number
        * @param error error flag if it is interested in the calling party
        * @return generated address
        * @remark one supported format is explicit string representation hex format "FFF1::FFF2::FFF3::FFF4::FFF5::FFF6::FFF7::FFF8"
        * @remark IPv6 addresses often include a sequence of zeros and, as a notational convenience, two colons '::' (only one or zero times) can be employed to indicate it.
        * @remark next supported format is wildcard address written as "::" "0::0" to the broadcast packet 
        * @remark next supported format is "0::0" or "::" or "NULL" - if you want to use any address (any Network Interface Card) for incoming connections.
        * @remark An IPv4-mapped IPv6 address has the following format -- ::FFFF:204.152.189.116
        */
        sockaddr_in6 getSocketAddressIPv6(const char* address, unsigned short port, bool* error) const;

        bool checkAlreadyCallInErrorCase() const;

        static std::string getTextDescription(sockaddr_in& address);
        static std::string getTextDescription(sockaddr_in6& address);

        /** Get maximum incoming buffer size in bytes
        */
        size_t getIncomigBufferSize();
        
        /** Set maximum incoming buffer size in bytes
        */
        bool setIncomigBufferSize(size_t incomingBufferSize);

    private:
        /** Get a reference to global variable which hold is network subsystem have been initialized or not
        */
        static bool& gNetworkSubSystemInitializedFlag();

    private:
        /** Create socket via system call
        *@return true socket have been successfully removed
        */
        bool createSocket(Protocol protocolType);

        /** Remove socket
        *@return true socket have been successfully removed
        */
        bool deleteSocket();

        /** Private constructor for internal use
        * @param protocol protocol for the socket
        * @param s raw socket descriptors
        */
        Socket(Protocol protocol, SOCKET s, const std::string& theAddressInfo);

        SOCKET socketObj;        ///< Socket object (handle to OS)
        Protocol protocolUse;    ///< Used transport protocol
        std::string addressInfo; ///< Some text description of address which bind to the socket
        bool isActive;           ///< Is socket active (i.e. it can be used to carry connect() ) or passive ( i.e. it can be used for accept() connection, but not for connect() )
    };
}
