import { Card, Form, InputNumber, Typography, Divider, Switch, Select } from 'antd'
import type { StrategySchema } from '@/types'

const { Text } = Typography

interface Props {
  value: StrategySchema
  onChange: (strategy: StrategySchema) => void
  maxDevices?: number
}

export default function StrategyForm({ value, onChange, maxDevices }: Props) {
  const totalParallelism = value.tp_degree * value.pp_degree * value.dp_degree * (value.sp_degree || 1)
  const isValid = maxDevices ? totalParallelism <= maxDevices : true

  return (
    <Card title="Parallel Strategy" style={{ marginBottom: 16 }}>
      <Form layout="vertical">
        <Form.Item label="Tensor Parallelism (TP)" required>
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            max={64}
            value={value.tp_degree}
            onChange={(v) => onChange({ ...value, tp_degree: v ?? 1 })}
          />
        </Form.Item>
        
        <Form.Item label="Pipeline Parallelism (PP)" required>
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            max={64}
            value={value.pp_degree}
            onChange={(v) => onChange({ ...value, pp_degree: v ?? 1 })}
          />
        </Form.Item>
        
        <Form.Item label="Data Parallelism (DP)" required>
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            max={1024}
            value={value.dp_degree}
            onChange={(v) => onChange({ ...value, dp_degree: v ?? 1 })}
          />
        </Form.Item>
        
        <Form.Item label="Sequence Parallelism (SP)" help="Ulysses/Ring Attention parallelism">
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            max={64}
            value={value.sp_degree || 1}
            onChange={(v) => onChange({ ...value, sp_degree: v ?? 1 })}
          />
        </Form.Item>
        
        <Form.Item label="Expert Parallelism (EP)" help="MoE expert parallelism">
          <InputNumber
            style={{ width: '100%' }}
            min={1}
            max={64}
            value={value.ep_degree || 1}
            onChange={(v) => onChange({ ...value, ep_degree: v ?? 1 })}
          />
        </Form.Item>
        
        <Form.Item label="Activation Checkpointing" help="Enable gradient checkpointing to save memory">
          <Switch
            checked={value.activation_checkpointing}
            onChange={(v) => onChange({ ...value, activation_checkpointing: v })}
          />
        </Form.Item>
        
        <Form.Item label="ZeRO Stage" help="ZeRO optimization stage">
          <Select
            style={{ width: '100%' }}
            value={value.zero_stage}
            onChange={(v) => onChange({ ...value, zero_stage: v })}
            options={[
              { value: 0, label: 'Stage 0 (Disabled)' },
              { value: 1, label: 'Stage 1 (Optimizer State Sharding)' },
              { value: 2, label: 'Stage 2 (+ Gradient Sharding)' },
              { value: 3, label: 'Stage 3 (+ Parameter Sharding)' }
            ]}
          />
        </Form.Item>
        
        <Divider />
        
        <div style={{ textAlign: 'center' }}>
          <Text strong>Total Parallelism: {totalParallelism}</Text>
          {maxDevices && (
            <Text type={isValid ? 'success' : 'danger'} style={{ marginLeft: 8 }}>
              / {maxDevices} devices
            </Text>
          )}
          {!isValid && (
            <Text type="danger" style={{ display: 'block', marginTop: 8 }}>
              Warning: Total parallelism exceeds device count
            </Text>
          )}
        </div>
      </Form>
    </Card>
  )
}