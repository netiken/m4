#ifndef _TYPE_
#define _TYPE_

#include <cstdint>
#include <list>
#include <memory>

/// Callback function pointer: "void func(void*)"
using Callback = void (*)(void*);

/// Callback function argument: void*
using CallbackArg = void*;

/// Device ID which starts from 0
using DeviceId = int;

/// Chunk size in Bytes
using ChunkSize = uint64_t;

/// Bandwidth in GB/s
using Bandwidth = double;

/// Latency in ns
using Latency = double;

/// Event time in ns
using EventTime = uint64_t;

/// Event ID
using EventId = uint64_t;

class Chunk;
class Link;
class Node;
class Topology;


using Route = std::list<std::shared_ptr<Node>>;

/// Basic multi-dimensional topology building blocks
enum class TopologyBuildingBlock { Undefined, Ring, FullyConnected, Switch };


#endif // _TYPE_
