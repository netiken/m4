#include "flow-size-tag.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (FlowSizeTag);

TypeId 
FlowSizeTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FlowSizeTag")
    .SetParent<Tag> ()
    .AddConstructor<FlowSizeTag> ()
  ;
  return tid;
}
TypeId 
FlowSizeTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
uint32_t 
FlowSizeTag::GetSerializedSize (void) const
{
  return 4;
}
void 
FlowSizeTag::Serialize (TagBuffer buf) const
{
  buf.WriteU32 (m_flowSize);
}
void 
FlowSizeTag::Deserialize (TagBuffer buf)
{
  m_flowSize = buf.ReadU32 ();
}
void 
FlowSizeTag::Print (std::ostream &os) const
{
  os << "FlowSize=" << m_flowSize;
}
FlowSizeTag::FlowSizeTag ()
  : Tag () 
{
}

FlowSizeTag::FlowSizeTag (uint32_t size)
  : Tag (),
    m_flowSize (size)
{
}

void
FlowSizeTag::SetFlowSize (uint32_t size)
{
  m_flowSize = size;
}
uint32_t
FlowSizeTag::GetFlowSize (void) const
{
  return m_flowSize;
}

} // namespace ns3

