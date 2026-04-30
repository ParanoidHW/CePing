import React from 'react'
import { Card, Form, InputNumber, Typography, Divider } from 'antd'
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
            onChange={(v) => onChange({ ...value, sp_degree: v })}
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