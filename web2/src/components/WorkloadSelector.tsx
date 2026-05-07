import { useEffect } from 'react'
import { Card, Select, Spin, Alert, Typography } from 'antd'
import { useWorkloads, useWorkloadSchema } from '@/hooks'
import type { WorkloadSchema } from '@/types'

const { Title, Text } = Typography

interface Props {
  value: string | null
  onNameChange: (name: string) => void
  onSchemaReady: (name: string, schema: WorkloadSchema) => void
}

export default function WorkloadSelector({ value, onNameChange, onSchemaReady }: Props) {
  const { categories, loading, error } = useWorkloads()
  const { schema, loading: schemaLoading } = useWorkloadSchema(value)

  useEffect(() => {
    if (schema && value) {
      onSchemaReady(value, schema)
    }
  }, [schema, value])

  if (error) {
    return <Alert type="error" message="Failed to load workloads" description={error} />
  }

  const options = Object.entries(categories).map(([category, workloads]) => ({
    label: category.toUpperCase(),
    options: workloads.map((w) => ({
      value: `${category}/${w}`,
      label: w
    }))
  }))

  return (
    <Card title="Workload Selection" style={{ marginBottom: 16 }}>
      <Spin spinning={loading}>
        <Select
          style={{ width: '100%' }}
          placeholder="Select a workload type"
          value={value}
          onChange={(v) => onNameChange(v)}
          options={options}
          showSearch
          filterOption={(input, option) =>
            (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
          }
        />
        
        {schemaLoading && <Spin style={{ marginTop: 16 }} />}
        
        {schema && (
          <div style={{ marginTop: 16 }}>
            <Title level={5}>{schema.display_name}</Title>
            <Text type="secondary">{schema.description}</Text>
            <div style={{ marginTop: 8 }}>
              <Text code>{schema.workload_type}</Text>
              <Text type="secondary"> | </Text>
              <Text code>{schema.compute_mode}</Text>
            </div>
            {schema.throughput_metric && (
              <Text type="secondary" style={{ marginTop: 4 }}>
                Metric: {schema.throughput_metric}
              </Text>
            )}
          </div>
        )}
      </Spin>
    </Card>
  )
}