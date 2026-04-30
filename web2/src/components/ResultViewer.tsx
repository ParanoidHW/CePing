import React from 'react'
import { Card, Table, Statistic, Row, Col, Alert, Typography, Descriptions, Divider, Progress, Tag } from 'antd'
import type { EvaluationResult, PerformanceResult, BreakdownResult } from '@/types'
import BreakdownTree from './BreakdownTree'

const { Title, Text } = Typography

interface Props {
  result: EvaluationResult | null
  loading: boolean
}

export default function ResultViewer({ result, loading }: Props) {
  if (loading) {
    return <Card loading={true} style={{ marginTop: 16 }} />
  }

  if (!result) {
    return null
  }

  if (!result.success) {
    return (
      <Card style={{ marginTop: 16 }}>
        <Alert
          type="error"
          title="Evaluation Failed"
          message="Validation errors occurred"
          description={
            <ul>
              {result.validation?.errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          }
        />
        {result.validation?.warnings.length > 0 && (
          <Alert
            type="warning"
            style={{ marginTop: 16 }}
            message="Warnings"
            description={
              <ul>
                {result.validation?.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            }
          />
        )}
      </Card>
    )
  }

  const perf = result.result!

  return (
    <Card style={{ marginTop: 16 }}>
      <Title level={4}>Evaluation Results</Title>
      
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Statistic
            title="Total Time"
            value={perf.total_time_sec}
            suffix="sec"
            precision={2}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Peak Memory"
            value={perf.peak_memory_gb}
            suffix="GB"
            precision={2}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title={perf.throughput_metric}
            value={perf.throughput}
            precision={0}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Workload Type"
            value={perf.workload_type}
            valueStyle={{ fontSize: 16 }}
          />
        </Col>
      </Row>
      
      {perf.metrics && Object.keys(perf.metrics).length > 0 && (
        <Card size="small" title="Metrics" style={{ marginBottom: 16 }}>
          <Descriptions column={3} size="small">
            {Object.entries(perf.metrics).map(([key, value]) => (
              <Descriptions.Item key={key} label={key}>
                {typeof value === 'number' ? value.toFixed(2) : value}
              </Descriptions.Item>
            ))}
          </Descriptions>
        </Card>
      )}
      
      {perf.stages && perf.stages.length > 0 && (
        <Card size="small" title="Stage Breakdown" style={{ marginBottom: 16 }}>
          <Table
            dataSource={perf.stages}
            rowKey="name"
            pagination={false}
            size="small"
            columns={[
              { title: 'Stage', dataIndex: 'name', key: 'name' },
              { title: 'Time (sec)', dataIndex: 'time_sec', key: 'time', render: (v) => v.toFixed(3) },
              { title: 'Memory (GB)', dataIndex: 'memory_gb', key: 'memory', render: (v) => v.toFixed(2) },
              {
                title: 'Percentage',
                key: 'percentage',
                render: (_, r) => (
                  <Progress
                    percent={(r.time_sec / perf.total_time_sec) * 100}
                    size="small"
                    format={(p) => `${p?.toFixed(1)}%`}
                  />
                )
              }
            ]}
          />
        </Card>
      )}
      
      {perf.breakdown && <BreakdownTree breakdown={perf.breakdown} />}
    </Card>
  )
}