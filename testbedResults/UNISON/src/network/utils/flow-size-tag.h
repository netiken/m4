#ifndef FLOW_SIZE_TAG_H
#define FLOW_SIZE_TAG_H

#include "ns3/tag.h"

namespace ns3 {

class FlowSizeTag : public Tag
{
public:
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer buf) const;
  virtual void Deserialize (TagBuffer buf);
  virtual void Print (std::ostream &os) const;
  FlowSizeTag ();
  FlowSizeTag (uint32_t flowSize);
  void SetFlowSize (uint32_t flowSize);
  uint32_t GetFlowSize (void) const;
private:
  uint32_t m_flowSize;
};

} // namespace ns3

#endif /* FLOW_SIZE_TAG_H */
