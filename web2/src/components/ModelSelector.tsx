import React from 'react'
import { Card, Select, Spin, Alert, Typography, Tag } from 'antd'
import { useModels, useModelSchema } from '@/hooks'
import type { ModelSchema } from '@/types'

const { Title, Text } = Typography

interface Props {
  workload: string | null
  value: string | null
  onChange: (model: string, schema: ModelSchema) => void
}

export default function ModelSelector({ workload, value, onChange }: Props) {
  const { models, loading, error } = useModels(workload?.split('/')[0])
  const { schema, loading: schemaLoading } = useModelSchema(value)

  React.useEffect(() => {
    if (schema && value) {
      onChange(value, schema)
    }
  }, [schema, value])

  if (error) {
    return <Alert type="error" message="Failed to load models" description={error} />
  }

  const options = models.map((m) => ({
    value: m.name,
    label: (
      <div>
        <span>{m.display_name}</span>
        <Tag style={{ marginLeft: 8 }} color="blue">{m.architecture}</Tag>
        {m.sparse_type !== 'dense' && (
          <Tag color="green">{m.sparse_type}</Tag>
        )}
      </div>
    )
  }))

  return (
    <Card title="Model Selection" style={{ marginBottom: 16 }}>
      <Spin spinning={loading}>
        <Select
          style={{ width: '100%' }}
          placeholder="Select a model"
          value={value}
          onChange={(v) => onChange(v, schema!)}
          options={options}
          showSearch
          filterOption={(input, option) =>
            (option?.value as string)?.toLowerCase().includes(input.toLowerCase())
          }
          disabled={!workload}
        />
        
        {schemaLoading && <Spin style={{ marginTop: 16 }} />}
        
        {schema && (
          <div style={{ marginTop: 16 }}>
            <Title level={5}>{schema.description}</Title>
            <div style={{ marginTop: 8 }}>
              <Tag color="blue">{schema.architecture}</Tag>
              {schema.sparse_type !== 'dense' && (
                <Tag color="green">{schema.sparse_type}</Tag>
              )}
            </div>
            {schema.attention_features.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <Text type="secondary">Features: </Text>
                {schema.attention_features.map((f) => (
                  <Tag key={f}>{f}</Tag>
                ))}
              </div>
            )}
          </div>
        )}
      </Spin>
    </Card>
  )
}