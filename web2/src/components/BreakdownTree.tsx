import React from 'react'
import { Card, Collapse, Table, Typography, Descriptions, Tag } from 'antd'
import type { BreakdownResult, MemoryBreakdown, CommunicationBreakdown } from '@/types'

const { Text } = Typography

interface Props {
  breakdown: BreakdownResult
}

export default function BreakdownTree({ breakdown }: Props) {
  const renderMemoryBreakdown = (mem: MemoryBreakdown) => (
    <Descriptions column={3} size="small">
      <Descriptions.Item label="Parameters">{mem.parameters_gb.toFixed(2)} GB</Descriptions.Item>
      <Descriptions.Item label="Activations">{mem.activations_gb.toFixed(2)} GB</Descriptions.Item>
      {mem.gradients_gb && (
        <Descriptions.Item label="Gradients">{mem.gradients_gb.toFixed(2)} GB</Descriptions.Item>
      )}
      {mem.optimizer_gb && (
        <Descriptions.Item label="Optimizer">{mem.optimizer_gb.toFixed(2)} GB</Descriptions.Item>
      )}
      {mem.kv_cache_gb && (
        <Descriptions.Item label="KV Cache">{mem.kv_cache_gb.toFixed(2)} GB</Descriptions.Item>
      )}
    </Descriptions>
  )

  const renderCommunicationBreakdown = (comm: CommunicationBreakdown) => (
    <Descriptions column={4} size="small">
      <Descriptions.Item label="AllReduce">{comm.all_reduce_gb.toFixed(3)} GB</Descriptions.Item>
      <Descriptions.Item label="AllGather">{comm.all_gather_gb.toFixed(3)} GB</Descriptions.Item>
      <Descriptions.Item label="ReduceScatter">{comm.reduce_scatter_gb.toFixed(3)} GB</Descriptions.Item>
      {comm.all_to_all_gb && (
        <Descriptions.Item label="AllToAll">{comm.all_to_all_gb.toFixed(3)} GB</Descriptions.Item>
      )}
    </Descriptions>
  )

  const items = []

  if (breakdown.by_stage) {
    items.push({
      key: 'stage',
      label: 'By Stage',
      children: (
        <Table
          dataSource={Object.entries(breakdown.by_stage).map(([name, data]) => ({
            name,
            ...data
          }))}
          rowKey="name"
          pagination={false}
          size="small"
          columns={[
            { title: 'Stage', dataIndex: 'name', key: 'name' },
            { title: 'Time (sec)', dataIndex: 'time_sec', key: 'time', render: (v) => v.toFixed(3) },
            { title: 'Memory (GB)', dataIndex: 'memory_gb', key: 'memory', render: (v) => v.toFixed(2) },
            { title: '%', dataIndex: 'percentage', key: 'pct', render: (v) => `${v.toFixed(1)}%` }
          ]}
        />
      )
    })
  }

  if (breakdown.by_phase) {
    items.push({
      key: 'phase',
      label: 'By Phase',
      children: (
        <Table
          dataSource={Object.entries(breakdown.by_phase).map(([key, data]) => ({
            key,
            ...data
          }))}
          rowKey="key"
          pagination={false}
          size="small"
          columns={[
            { title: 'Phase', dataIndex: 'name', key: 'name' },
            { title: 'Time (sec)', dataIndex: 'time_sec', key: 'time', render: (v) => v.toFixed(3) },
            { title: 'Memory (GB)', dataIndex: 'memory_gb', key: 'memory', render: (v) => v.toFixed(2) }
          ]}
        />
      )
    })
  }

  if (breakdown.by_submodule) {
    items.push({
      key: 'submodule',
      label: 'By Submodule',
      children: (
        <Table
          dataSource={Object.entries(breakdown.by_submodule).map(([key, data]) => ({
            key,
            ...data
          }))}
          rowKey="key"
          pagination={false}
          size="small"
          columns={[
            { title: 'Submodule', dataIndex: 'name', key: 'name' },
            { title: 'Type', dataIndex: 'type', key: 'type', render: (v) => <Tag>{v}</Tag> },
            { title: 'Time (sec)', dataIndex: 'time_sec', key: 'time', render: (v) => v.toFixed(4) },
            { title: 'Memory (GB)', dataIndex: 'memory_gb', key: 'memory', render: (v) => v.toFixed(3) },
            { title: 'FLOPs', dataIndex: 'flops', key: 'flops', render: (v) => v ? v.toFixed(0) : '-' }
          ]}
        />
      )
    })
  }

  if (breakdown.memory) {
    items.push({
      key: 'memory',
      label: 'Memory Breakdown',
      children: renderMemoryBreakdown(breakdown.memory)
    })
  }

  if (breakdown.communication) {
    items.push({
      key: 'communication',
      label: 'Communication Breakdown',
      children: renderCommunicationBreakdown(breakdown.communication)
    })
  }

  if (items.length === 0) {
    return null
  }

  return (
    <Card size="small" title="Detailed Breakdown" style={{ marginTop: 16 }}>
      <Collapse items={items} defaultActiveKey={['stage', 'memory']} />
    </Card>
  )
}