#include "burt/system/include/network/Socket.h"
#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include "burt/copylocal/include/Data.h"
#include "burt/copylocal/include/MutableData.h"

#include <string.h>
#include <errno.h>
#include <iostream>
#include <assert.h>

namespace
{
    int lastErrorCode()
    {
#if BURT_WINDOWS
        return WSAGetLastError();
#else
        return errno;
#endif
    }

}
namespace burt
{
    bool& Socket::gNetworkSubSystemInitializedFlag() 
    {
        static bool isInit = false;
        return isInit;
    }

    bool Socket::isNetworkSubSystemInitialized()
    {
        return gNetworkSubSystemInitializedFlag();
    }

    bool Socket::initNetworkSubSystem()
    {
#if BURT_WINDOWS
        WORD wVersionRequested = MAKEWORD(2, 2);
        WSADATA sVendorInfo = {};

        // https://msdn.microsoft.com/ru-ru/library/windows/desktop/ms741394(v=vs.85).aspx
        // Initiates use of WS2_32.DLL by a process.
        if (WSAStartup(wVersionRequested, &sVendorInfo) != 0)
        {
            std::cout << "Windows Sockets 2. Error WSAStartup. Code: " << lastErrorCode() << '\n';
            return false;
        }
#else
        ;
#endif
        gNetworkSubSystemInitializedFlag() = true;
        return true;
    }

    void Socket::deinitNetworkSubSystem()
    {
#if BURT_WINDOWS
        WSACleanup();
#else
        ;
#endif
        gNetworkSubSystemInitializedFlag() = false;
    }

    sockaddr_in Socket::getSocketAddressIPv4(const char * address, unsigned short port, bool * error) const
    {
        if (error != nullptr)
            *error = false;

        if (address == nullptr)
        {
            // INADDR_ANY -- wildcard address. Allow to recevie datagrams from any interface
            address = "0.0.0.0";
        }
        else if (strcmp(address, "localhost") == 0)
        {
            // Loopback address
            address = "127.0.0.1";
        }

        // The sockaddr_in structure specifies a transport address and port for the AF_INET (IPV4) address family
        sockaddr_in res = {};
        res.sin_family = AF_INET;

        // To use "inet_pton" system call the address should be "ddd.ddd.ddd.ddd", where ddd is a decimal number of up to three digits in the range 0 to 255.
        // inet_pton() returns 1 on success
        if (inet_pton(res.sin_family, address, &res.sin_addr) == 1)
        {
            // Convert host 16 bits nubmer host byte order to network byte order. 
            // And Network byte order is big-endian (https://www.rfc-editor.org/rfc/rfc1700)
            res.sin_port = htons(port);
            return res;
        }
        else
        {
            //INADDR_ANY
            addrinfo* result = nullptr;
            addrinfo addrHints = {};
            addrHints.ai_family = AF_INET;

            // The following call may sends a request via DNS
            if (int res = getaddrinfo(address, nullptr, nullptr, &result); res != 0)
            {
                std::cout << "getaddrinfo failed. Code: " << res << " [" << gai_strerror(res) << "]" << '\n';
                if (error)
                    *error = true;
                
                sockaddr_in resDefault = {};
                return resDefault;
            }
            
            // getaddrinfo() returns 0 if it succeeds
            bool found = false;
            for (addrinfo* a = result; a != nullptr && found == false; a = a->ai_next)
            {
                if (result->ai_family == AF_INET)
                {
                    // AF_INET -- Internet address family formats for IPv4 
                    res = *((sockaddr_in*)a->ai_addr);
                    assert(sizeof(sockaddr_in) == a->ai_addrlen);
                    
                    res.sin_family = result->ai_family;
                    // Convert host 16 bits nubmer host byte order to network byte order. 
                    // And Network byte order is big-endian (https://www.rfc-editor.org/rfc/rfc1700)
                    res.sin_port = htons(port);
                    found = true;
                }
            }

            freeaddrinfo(result);
            
            if (!found && error)
            {
                *error = true;
            }
        }

        return res;
    }

    sockaddr_in6 Socket::getSocketAddressIPv6(const char* address, unsigned short port, bool* error) const
    {
        if (error != NULL)
            *error = false;

        if (address == nullptr)
        {
            // IPv6 equivalents for wildcard address
            address = "0::0";
        }
        else if (strcmp(address, "localhost") == 0)
        {
            // IPv6 loopback address
            address = "::1";
        }

        sockaddr_in6 res = {};
        res.sin6_family = AF_INET6;

        if (inet_pton(res.sin6_family, address, &res.sin6_addr) == 1)
        {
            // inet_pton() returns 1 on success
            
            res.sin6_scope_id = 0;                // RFC-3493 and RFC-4007 provide information about sin6_scope_id.
            res.sin6_flowinfo = 0;                // RFC-2460 and RFC-3697 provide information about IPv6 flow control.
            res.sin6_port = htons(port); // Convert host 16 bits nubmer host byte order to network byte order. Network byte order is big-endian.

            return res;
        }
        else
        {
            addrinfo* result = nullptr;
            addrinfo addrHints = {};
            addrHints.ai_family = AF_INET6;
                
            // The following call may sens a request via DNS
            int resCode = getaddrinfo(address, nullptr, &addrHints, &result);
            assert(resCode == 0);

            if (resCode != 0)
            {
                std::cout << "getaddrinfo failed. Code: " << resCode << " [" << gai_strerror(resCode) << "]" << '\n';
                if (error)
                    *error = true;

                sockaddr_in6 resDefault = {};
                return resDefault;
            }


            bool found = false;
            for (addrinfo* a = result; a != nullptr && found == false; a = a->ai_next)
            {
                if (result->ai_family == AF_INET6)
                {
                    res = *((sockaddr_in6*)a->ai_addr);
                    assert(sizeof(sockaddr_in6) == a->ai_addrlen);

                    res.sin6_family = result->ai_family;
                    res.sin6_scope_id = 0;       // RFC-3493 and RFC-4007 provide information about sin6_scope_id.
                    res.sin6_flowinfo = 0;       // RFC-2460 and RFC-3697 provide information about IPv6 flow control.
                    res.sin6_port = htons(port); // Convert host 16 bits nubmer host byte order to network byte order. Network byte order is big-endian.

                    found = true;
                }
            }

            freeaddrinfo(result);
            
            if (!found && error)
            {
                *error = true;
            }
        }

        return res;
    }

    Socket::Socket()
    : socketObj(INVALID_SOCKET)
    , isActive(true)
    , protocolUse(Protocol::UNKNOWN)
    {
    }

    Socket::Socket(Socket::Protocol protocol)
    : socketObj(INVALID_SOCKET)
    , isActive(true)
    , protocolUse(protocol)
    {
        socketObj = INVALID_SOCKET;
        if (createSocket(protocol) == false)
        {
            std::cout << "Can not create socket\n";
            assert(!"Can not create socket");
        }
    }

    bool Socket::connect(const char * address, unsigned short port)
    {
        // An active open of a socket in order to establish a connection 
        if (isActive == false)
        {
            std::cout << "Socket is not in active state. It can not be used to establish connection\n";
            return false;
        }
        
        // Perform Active Open with 3-way handshaking
        //  - Client sends SYN to server with intial seq. number
        //  - Server response with SYN+ACK typically
        //  - Finally client sends ACK to server
        
        bool error = false;
           
        if (protocolUse == Protocol::TCPv4 || protocolUse == Protocol::UDPv4)
        {
            sockaddr_in addr = getSocketAddressIPv4(address, port, &error);
            static_assert(sizeof(sockaddr_in) == sizeof(sockaddr));
            
            if (error)
                return false;

            if (::connect(socketObj, (sockaddr*)&addr, sizeof(addr)) != 0)
            {
                std::cout << "Cannot perform connection. Error code: " << lastErrorCode() << '\n';
                return checkAlreadyCallInErrorCase();
            }
        }
        else if (protocolUse == Protocol::TCPv6 || protocolUse == Protocol::UDPv6)
        {
            sockaddr_in6 addr = getSocketAddressIPv6(address, port, &error);            
            if (error)
                return false;
            
            if (::connect(socketObj, (sockaddr*)&addr, sizeof(addr)) != 0)
            {
                std::cout << "Cannot perform connection. Error code: " << lastErrorCode() << '\n';
                return checkAlreadyCallInErrorCase();
            }
        }

        return true;
    }

    bool Socket::bind(const char * address, unsigned short port, bool reuseAddress)
    {
        bool error = false;

        if (protocolUse == Protocol::TCPv4 || protocolUse == Protocol::UDPv4)
        {
            sockaddr_in addr = getSocketAddressIPv4(address, port, &error);
            static_assert(sizeof(sockaddr_in) == sizeof(sockaddr));

            if (error)
                return false;

            addressInfo = getTextDescription(addr);

            if (reuseAddress)
            {
                int optval = 1;
                if (setsockopt(socketObj, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval, sizeof(optval)) == -1)
                {
                    std::cout << "Cannot set SO_REUSEADDR option. Error code: " << lastErrorCode() << '\n';
                    return false;
                }
            }

            if (::bind(socketObj, (sockaddr*)&addr, sizeof(addr)) != 0)
            {
                return checkAlreadyCallInErrorCase();
            }
        }
        else if (protocolUse == Protocol::TCPv6 || protocolUse == Protocol::UDPv6)
        {
            sockaddr_in6 addr = getSocketAddressIPv6(address, port, &error);
            if (error)
                return false;
            
            addressInfo = getTextDescription(addr);

            if (reuseAddress)
            {
                int optval = 1;
                if (setsockopt(socketObj, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval, sizeof(optval)) == -1)
                {
                    std::cout << "Cannot set SO_REUSEADDR option. Error code: " << lastErrorCode() << '\n';
                    return false;
                }
            }

            if (::bind(socketObj, (sockaddr*)&addr, sizeof(addr)) != 0)
            {
                return checkAlreadyCallInErrorCase();
            }
        }

        return true;
    }

    bool Socket::bind(unsigned short port, bool reuseAddress)
    {
        return bind(nullptr, port, reuseAddress);
    }

    bool Socket::listen(int maxPendingConnections)
    {
        // A passive open of a socket
        bool res = (::listen(socketObj, maxPendingConnections) == 0);
        if (res == false)
        {
            res = checkAlreadyCallInErrorCase();
            return res;
        }
        else
        {
            isActive = false;
            return res;
        }        
    }

    std::unique_ptr<Socket> Socket::serverAcceptConnection()
    {
        if (isActive == true)
        {
            std::cout << "Socket is in active state. It can not be used to accept connection. Please call listen first.\n";
            return std::unique_ptr<Socket>(nullptr);
        }

        if (protocolUse == Protocol::TCPv4 || protocolUse == Protocol::UDPv4)
        {
            sockaddr_in addr = {};
            socklen_t len = sizeof(addr);
            SOCKET incomingConnect = ::accept(getHandleFromOS(), (sockaddr*) &addr, &len);
         
            if (incomingConnect == INVALID_SOCKET)
                return std::unique_ptr<Socket>(nullptr);

            std::unique_ptr<Socket> res = std::unique_ptr<Socket>( new Socket(protocolUse, incomingConnect, getTextDescription(addr)) );
            return res;
        }
        else if (protocolUse == Protocol::TCPv6 || protocolUse == Protocol::UDPv6)
        {
            sockaddr_in6 addr = {};
            socklen_t len = sizeof(addr);
            SOCKET incomingConnect = ::accept(getHandleFromOS(), (sockaddr*)&addr, &len);

            if (incomingConnect == INVALID_SOCKET)
                return std::unique_ptr<Socket>(nullptr);

            std::unique_ptr<Socket> res = std::unique_ptr<Socket>( new Socket(protocolUse, incomingConnect, getTextDescription(addr)) );
            
            return res;
        }
        else
        {
            assert(!"UKNOWN PROTOCOL");
            return std::unique_ptr<Socket>(nullptr);
        }
    }

    Socket::Socket(Protocol protocol, SOCKET s, const std::string& theAddressInfo)
    : socketObj(s)
    , isActive(true)
    , protocolUse(protocol)
    , addressInfo(theAddressInfo)
    {
    }

    Socket::~Socket() {
        deleteSocket();
    }

    Socket::Socket(Socket&& rhs) noexcept
    : socketObj(rhs.socketObj)
    , protocolUse(rhs.protocolUse)
    , addressInfo(rhs.addressInfo)
    , isActive(rhs.isActive)
    {
        rhs.socketObj = INVALID_SOCKET;
    }

    Socket& Socket::operator = (Socket&& rhs) noexcept
    {
        if (this == &rhs)
            return *this;
        
        deleteSocket();

        socketObj = rhs.socketObj;
        protocolUse = rhs.protocolUse;
        addressInfo = rhs.addressInfo;
        isActive = rhs.isActive;

        rhs.socketObj = INVALID_SOCKET;
        rhs.addressInfo.clear();

        return *this;
    }

    Socket::Protocol Socket::getProtocol() const {
        return protocolUse;
    }

    bool Socket::createSocket(Socket::Protocol protocolType)
    {
        int protocol = -1;
        int addressFamily = -1;

        switch (protocolType)
        {
        case Protocol::TCPv4:
            addressFamily = AF_INET; // AF_INET communication domain allows communication between applications running on hosts connected via an Internet via usuing IPV4
            protocol = SOCK_STREAM;
            break;
        case Protocol::UDPv4:
            addressFamily = AF_INET; // AF_INET communication domain allows communication between applications running on hosts connected via an Internet via usuing IPV4
            protocol = SOCK_DGRAM;
            break;
        case Protocol::TCPv6:
            addressFamily = AF_INET6; // AF_INET communication domain allows communication between applications running on hosts connected via an Internet via usuing IPV6
            protocol = SOCK_STREAM;
            break;
        case Protocol::UDPv6:
            addressFamily = AF_INET6; // AF_INET communication domain allows communication between applications running on hosts connected via an Internet via usuing IPV6
            protocol = SOCK_DGRAM;
            break;
        case Protocol::UNKNOWN:
            addressFamily = 0;        // Some default values for internet sccket
            protocol = 0;
            break;
        }

        // sockaddr_in -- is a datastructure to use for AF_INET/IPV4 to specify address:
        // 32-bit IPv4 address 
        // 16-bit port number
        // 
        // sockaddr_in6 -- is a datastructure to use for AF_INET6/IPV6 to specify address
        // 128-bit IPv6 address
        // 16-bit port number
        //
        protocolUse = protocolType;

#if BURT_WINDOWS
        // Overlapped sockets can utilize WSASend, WSASendTo, WSARecv, WSARecvFrom, and WSAIoctl 
        // for overlapped I/O operations, which allow 
        // multiple operations to be initiated and in progress simultaneously.

        // Create new socket system object
        if ((socketObj = WSASocket(addressFamily, protocol, 0, NULL, 0, WSA_FLAG_OVERLAPPED)) == INVALID_SOCKET)
        {
            std::cout << "Windows Sockets 2. Error create socket. Code: " << lastErrorCode() << '\n';
            return false;
        }

#else
        // socket() is system call, which returns a file descriptor used to refer to the socket in subsequent system calls

        // Create new socket system object
        if ((socketObj = socket(addressFamily, protocol, 0)) == INVALID_SOCKET)
        {
            std::cout << "Posix Socket. Error create socket. Code: " << lastErrorCode() << '\n';
            return false;
        }
#endif

        return true;
    }

    bool Socket::deleteSocket()
    {
        // Close socket
        //  - Send FIN to another side
        //  - Another side sends ACK
        //  - Another side closes connection
        //  - Send FIN from another side to this side
        //  - Client response with ACK about closing

        if (socketObj != INVALID_SOCKET)
        {
#if BURT_WINDOWS
            bool deleteResult = (closesocket(socketObj) == 0);
#else
            bool deleteResult = (close(socketObj) == 0);
#endif
            socketObj = INVALID_SOCKET;

            return deleteResult;
        }

        return true;
    }

    void Socket::shutDownReceiveChannel() const
    {
        if (protocolUse == Protocol::TCPv4 || protocolUse == Protocol::TCPv6)
            shutdown(getHandleFromOS(), SD_RECEIVE);
    }

    void Socket::shutDownSendChannel() const
    {
        if (protocolUse == Protocol::TCPv4 || protocolUse == Protocol::TCPv6)
            shutdown(getHandleFromOS(), SD_SEND);
    }

    bool Socket::sendData(const void* buff, int len)
    {
        if (len == 0)
            return true;

        int total = 0;
        int n = 0;
        const unsigned char* byteBuffer = static_cast<const unsigned char*>(buff);

        while (total < len)
        {
            n = ::send(socketObj, (char*)(byteBuffer + total), len - total, 0);

            if (n == SOCKET_ERROR || n == 0)
            {
                return false;
            }

            total += n;
            // Buffer [buff, buf + total) is partially send
            //  If you need to be notified ASAP, make a callback potentiall from this point
        }

        return true;
    }

    bool Socket::recvData(void* buff, int len)
    {
        if (len == 0)
            return true;

        int total = 0;
        int n = 0;
        const unsigned char* byteBuffer = static_cast<const unsigned char*>(buff);

        while (total < len)
        {
            n = ::recv(socketObj, (char*)(byteBuffer + total), len - total, 0);

            // Connection has been closed
            if (n == 0)
            {
                std::cout << "Connection has been closed\n";
                return false;
            }

#if BURT_WINDOWS
            if (n == SOCKET_ERROR)
            {
                if (WSAGetLastError() == WSAETIMEDOUT)
                {
                    // Timeout
                    continue;
                }
                else
                {
                    // Another actual error
                    return false;
                }
            }
#else
            if (n == -1)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    // Timeout
                    continue;
                }
                else
                {
                    return false;
                }
            }
#endif
            total += n;
            
            // Buffer [buff, buf + total) is partially ready
            //  If you need to accept data asap, make a callback potentiall from this point
        }

        return true;
    }

    uint64_t Socket::getUint64()
    {
        uint64_t result = 0;
        bool recvResult = recvData(&result, sizeof(result));
        assert(recvResult == true);
        return result;
    }

    uint32_t Socket::getUint32()
    {
        uint32_t result = 0;
        bool recvResult = recvData(&result, sizeof(result));
        assert(recvResult == true);
        return result;
    }
    
    uint16_t Socket::getUint16()
    {
        uint16_t result = 0;
        bool recvResult = recvData(&result, sizeof(result));
        assert(recvResult == true);
        return result;
    }

    uint8_t Socket::getUint8()
    {
        uint8_t result = 0;
        bool recvResult = recvData(&result, sizeof(result));
        assert(recvResult == true);
        return result;
    }

    bool Socket::checkAlreadyCallInErrorCase() const
    {
        int err = 0;
        socklen_t lenOfError = sizeof(err);
        getsockopt(socketObj, SOL_SOCKET, SO_ERROR, (char*)&err, &lenOfError);

#if BURT_WINDOWS
        if (err == WSAEALREADY || err == WSAEINPROGRESS)
            return true;
#else
        if (err == EALREADY || err == EINPROGRESS)
            return true;
#endif
        return false;
    }

    std::string Socket::getTextDescription(sockaddr_in& address)
    {
        char hostName[NI_MAXHOST] = {};
        if (int res = getnameinfo((sockaddr*)&address, sizeof(address), hostName, sizeof(hostName), nullptr, 0, NI_NUMERICHOST); res != 0)
        {
            std::cout << "getnameinfo failed. Code: " << res << " [" << gai_strerror(res) << "]" << '\n';
            return std::string(hostName);            
        }
        return std::string(hostName);
    }

    std::string Socket::getTextDescription(sockaddr_in6& address)
    {
        char hostName[NI_MAXHOST] = {};
        if (int res = getnameinfo((sockaddr*)&address, sizeof(address), hostName, sizeof(hostName), nullptr, 0, NI_NUMERICHOST); res != 0)
        {
            std::cout << "getnameinfo failed. Code: " << res << " [" << gai_strerror(res) << "]" << '\n';
            return std::string(hostName);
        }
        return std::string(hostName);
    }


    SOCKET Socket::getHandleFromOS() const {
        return socketObj;
    }

    const std::string& Socket::getAddressInfo() const {
        return addressInfo;
    }
    
    size_t Socket::getAvailableBytesForRead() const {
#if BURT_WINDOWS
        u_long bytes = 0;
        if (ioctlsocket(socketObj, FIONREAD, (u_long*)&bytes) == 0)
        {
            return size_t(bytes);
        }
        else
        {
            assert(!"ERROR DURING REQUEST NUMBER OF BYTES IN SOCKET");
            return 0;
        }
#else
        int bytes_available = 0;
        if (ioctl(socketObj, FIONREAD, &bytes_available) == -1)
        {
            assert(!"ERROR DURING REQUEST NUMBER OF BYTES IN SOCKET");
            return 0;
        }
        else
        {
            return size_t(bytes_available);
        }
#endif
    }

    bool Socket::isSocketActive() const {
        return isActive;
    }

    bool Socket::setRecvBlockTimeout(unsigned int milliseconds)
    {
        if (socketObj == INVALID_SOCKET)
            return false;

#if BURT_WINDOWS
        // https://msdn.microsoft.com/ru-ru/library/windows/desktop/ms740476(v=vs.85).aspx
        DWORD val = milliseconds;

#else
        // http://man7.org/linux/man-pages/man7/socket.7.html
        timeval val = {};
        val.tv_sec = 0;                    // 0 seconds
        val.tv_usec = milliseconds * 1000; // ms to us
#endif
        return setsockopt(socketObj, SOL_SOCKET, SO_RCVTIMEO, (const char *)&val, sizeof(val)) == 0;
    }
    
    size_t Socket::getIncomigBufferSize()
    {
        if (socketObj == INVALID_SOCKET)
            return 0;

        int rcvBufferSize = 0;
        socklen_t sockOptSize = sizeof(rcvBufferSize);

        int res = getsockopt(socketObj, SOL_SOCKET, SO_RCVBUF, (char*)&rcvBufferSize, &sockOptSize);
        assert(res == 0);
        return size_t(rcvBufferSize);
    }
    
    bool Socket::setIncomigBufferSize(size_t incomingBufferSize)
    {
        if (socketObj == INVALID_SOCKET)
            return false;

        int incomingBufferSizeInt = int(incomingBufferSize);
        socklen_t sockOptSize = sizeof(incomingBufferSizeInt);
        return setsockopt(socketObj, SOL_SOCKET, SO_RCVBUF, (char*)&incomingBufferSizeInt, sockOptSize) == 0;
    }

    bool Socket::setNoDelay(bool noDelay)
    {
        {
#if BURT_WINDOWS
            // https://learn.microsoft.com/en-us/windows/win32/winsock/ipproto-tcp-socket-options
            DWORD flag = noDelay ? 1 /*Disable the Nagle algorithm, segments are always sent as soon as possible. */ : 0;
#else
            int flag = noDelay ? 1 /*Disable the Nagle algorithm, segments are always sent as soon as possible. */ : 0;
#endif
            if (setsockopt(socketObj, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag)) != 0)
            {
                return false;
            }
        }
        
        return true;
    }

    size_t waitForAvailableReadSocket(const Socket* socketsList,
                                    bool* dataIsReady4Socket, 
                                    size_t kSockets, 
                                    long microseconds2wait)
    {
        size_t readySockets = 0;
        
        // reset result
        memset(dataIsReady4Socket, 0, sizeof(bool) * kSockets);

        // set of file descriptors to be tested to see if input is possible
        fd_set readfds;
        SOCKET maxSocketHandle;
        timeval timeout = {};        

        constexpr size_t kMaxCheckConnections = FD_SETSIZE;        
        size_t processByBatches = burt::roundToNearestMultipleDown<kMaxCheckConnections>(kSockets);

        size_t i = 0;
        
        for (; i < processByBatches; i += kMaxCheckConnections)
        {
            FD_ZERO(&readfds);
            maxSocketHandle = SOCKET();
            
            for (size_t j = 0; j < kMaxCheckConnections; ++j)
            {
                // Adds the file descriptor <1st arg> to the set pointed to by <2nd arg>
                FD_SET(socketsList[i + j].getHandleFromOS(), &readfds);
                maxSocketHandle = std::max(maxSocketHandle, socketsList[i + j].getHandleFromOS());
            }
            
            timeout.tv_sec = 0;
            timeout.tv_usec = microseconds2wait;
            
            // 1st arg in select -- ndfs must be set one greater than the highest file descriptor number included in any of the three file descriptor sets
            // 2nd, 3rd, 4th -- read, write, exception ready scokets
            int selectResult = ::select(maxSocketHandle + 1, &readfds, nullptr, nullptr, &timeout);

            if (selectResult == 0)
            {
                // timeout
            }
            else if (selectResult == SOCKET_ERROR)
            {
                // A return value of �1 indicates that an error occurred
                // error
                assert(!"ERROR DURING SELECT");
                std::cout << "Error during select. Error code: " << lastErrorCode() << "n";
                return false;
            }
            else
            {
                // selectResult contains number of set sockets.
                // even more specific select() returns the total number of file descriptors marked as ready in all three returned set.
                // potentially socket handle can come into several sets                

                readySockets += selectResult;
                
                for (size_t j = 0; j < kMaxCheckConnections; ++j)
                {
                    bool isSet = FD_ISSET(socketsList[i + j].getHandleFromOS(), &readfds);
                    dataIsReady4Socket[i + j] = isSet;
                }
            }
        }
        
        // Process the rest
        {
            size_t theRest = kSockets - i;
            
            FD_ZERO(&readfds);
            maxSocketHandle = SOCKET();

            for (size_t j = i; j < kSockets; ++j)
            {
                FD_SET(socketsList[i + j].getHandleFromOS(), &readfds);
                maxSocketHandle = std::max(maxSocketHandle, socketsList[i + j].getHandleFromOS());
            }
            
            timeout.tv_sec = 0;
            timeout.tv_usec = microseconds2wait;
            
            int selectResult = ::select(maxSocketHandle + 1, &readfds, nullptr, nullptr, &timeout);

            if (selectResult == 0)
            {
                // timeout
            }
            else if (selectResult == SOCKET_ERROR)
            {
                // error
                assert(!"ERROR DURING SELECT");
                std::cout << "Error during select. Error code: " << lastErrorCode() << "n";
                return false;
            }
            else
            {
                readySockets += selectResult;
                // selectResult contains number of set sockets            
                for (size_t j = i; j < kSockets; ++j)
                {
                    bool isSet = FD_ISSET(socketsList[i + j].getHandleFromOS(), &readfds);
                    dataIsReady4Socket[i + j] = isSet;
                }
            }
        }

        return readySockets;
    }
}
