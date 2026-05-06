import { Card, Form, Select, InputNumber, Spin, Alert } from 'antd'
import { useHardware, useTopologies } from '@/hooks'
import type { HardwareSchema } from '@/types'

interface Props {
  value: HardwareSchema
  onChange: (hardware: HardwareSchema) => void
}

export default function HardwareForm({ value, onChange }: Props) {
  const { devices, loading: deviceLoading, error: deviceError } = useHardware()
  const { topologies, loading: topoLoading } = useTopologies()

  if (deviceError) {
    return <Alert type="error" message="Failed to load hardware" description={deviceError} />
  }

  const deviceOptions = Object.entries(devices).flatMap(([vendor, items]) => [
    { label: vendor.toUpperCase(), options: [] },
    ...items.map((d) => ({
      value: d.name,
      label: `${d.name} (${d.memory_gb}GB, ${d.tflops} TFLOPS)`
    }))
  ])

  const topologyOptions = topologies.map((t) => ({
    value: t.name,
    label: `${t.name} (${t.bandwidth_gb_s} GB/s)`
  }))

  return (
    <Card title="Hardware Configuration" style={{ marginBottom: 16 }}>
      <Spin spinning={deviceLoading || topoLoading}>
        <Form layout="vertical">
          <Form.Item label="Device Type" required>
            <Select
              style={{ width: '100%' }}
              value={value.device_preset}
              onChange={(v) => onChange({ ...value, device_preset: v })}
              options={deviceOptions}
              showSearch
              placeholder="Select device type"
            />
          </Form.Item>
          
          <Form.Item label="Number of Devices" required>
            <InputNumber
              style={{ width: '100%' }}
              min={1}
              max={1024}
              value={value.num_devices}
              onChange={(v) => onChange({ ...value, num_devices: v ?? 1 })}
            />
          </Form.Item>
          
          <Form.Item label="Topology" help="Inter-device communication topology">
            <Select
              style={{ width: '100%' }}
              value={value.topology_type}
              onChange={(v) => onChange({ ...value, topology_type: v })}
              options={topologyOptions}
              allowClear
              placeholder="Select topology (optional)"
            />
          </Form.Item>
        </Form>
      </Spin>
    </Card>
  )
}