import { Card, Form, InputNumber, Input, Switch } from 'antd'
import type { ParamSchema } from '@/types'

interface Props {
  title: string
  parameters: Record<string, ParamSchema>
  values: Record<string, number | string | boolean>
  onChange: (key: string, value: number | string | boolean) => void
}

export default function DynamicForm({ title, parameters, values, onChange }: Props) {
  const renderField = (key: string, schema: ParamSchema) => {
    const value = values[key] ?? (schema.default ?? '')

    switch (schema.type) {
      case 'number':
        return (
          <InputNumber
            style={{ width: '100%' }}
            value={value as number}
            onChange={(v) => onChange(key, v ?? (schema.default ?? 0))}
            min={schema.min}
            max={schema.max}
            placeholder={schema.description}
          />
        )
      
      case 'string':
        return (
          <Input
            value={value as string}
            onChange={(e) => onChange(key, e.target.value)}
            placeholder={schema.description}
          />
        )
      
      case 'boolean':
        return (
          <Switch
            checked={value as boolean}
            onChange={(v) => onChange(key, v)}
          />
        )
      
      default:
        return null
    }
  }

  if (!parameters || Object.keys(parameters).length === 0) {
    return null
  }

  return (
    <Card title={title} style={{ marginBottom: 16 }}>
      <Form layout="vertical">
        {Object.entries(parameters).map(([key, schema]) => (
          <Form.Item
            key={key}
            label={schema.label}
            required={schema.required}
            help={schema.description}
          >
            {renderField(key, schema)}
          </Form.Item>
        ))}
      </Form>
    </Card>
  )
}